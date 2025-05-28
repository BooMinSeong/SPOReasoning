import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Any, Optional
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn
import torch.nn.functional as F # _get_log_probs 내에서 F.log_softmax 사용 가정
from typing import Dict, List # Dict는 batch 타입 어노테이션, List는 mu_weights_k 타입 어노테이션에 사용

class SPOLoss(nn.Module):
    def __init__(self, alpha: float = 0.001, beta: float = 0.01, reference_model: nn.Module = None):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be greater than 0.")
        self.alpha = alpha
        self.beta = beta
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval() # 참조 모델은 평가 모드로 설정

    def _get_log_probs(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       labels: torch.Tensor) -> torch.Tensor:
        """
        Helper to get log probabilities of sequences from the model.
        Assumes `model` returns logits and `labels` are the target sequence (prompt masked).
        This calculates log_prob(response | prompt)
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask) # labels=None 명시 안해도 됨
        logits = outputs.logits # (batch_size_flat, sequence_length, vocab_size)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        log_probs_all_tokens = F.log_softmax(shift_logits, dim=-1) # (batch_size_flat, sequence_length-1, vocab_size)
        
        # Gather the log_probs for the actual next tokens
        # NLLLoss(log_softmax(inputs), target) 와 동일 효과: -log_probs.gather(dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
        # shift_labels이 -100인 부분은 무시되어야 함.
        
        loss_fct = nn.NLLLoss(ignore_index=-100, reduction='none') # per-token loss
        
        # NLLLoss expects (N, C) and target (N)
        # log_probs_all_tokens: (batch_flat, seq_len-1, vocab_size) -> (batch_flat * (seq_len-1), vocab_size)
        # shift_labels: (batch_flat, seq_len-1) -> (batch_flat * (seq_len-1))
        per_token_neg_log_likelihood = loss_fct(
            log_probs_all_tokens.view(-1, log_probs_all_tokens.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Reshape back to (batch_flat, seq_len - 1)
        per_token_neg_log_likelihood = per_token_neg_log_likelihood.view(labels.size(0), labels.size(1) - 1)
        
        # Sum of negative log likelihood where labels are not -100
        # 마스크된 부분(-100)은 NLLLoss에서 0으로 처리되므로, sum은 유효 토큰에 대한 합계가 됨.
        sequence_log_probs = -per_token_neg_log_likelihood.sum(dim=-1) # (batch_flat,)

        return sequence_log_probs

    def _calculate_base_term(self, chosen_log_prob: torch.Tensor, candidate_log_probs: torch.Tensor) -> torch.Tensor:
        log_ratio = (chosen_log_prob * self.alpha) - torch.logsumexp(candidate_log_probs * self.alpha, dim=-1)
        return log_ratio

    def ranked_preference_loss(self,
                               policy_model: nn.Module,
                               batch: Dict[str, torch.Tensor]
                               ) -> torch.Tensor:
        """
        Calculates the Soft Preference Optimization (SPO) loss for Ranked Preference Data.
        Includes a simplified DPO-like KL regularization based on the top-ranked response.
        """
        candidate_input_ids = batch['candidate_input_ids']    # (batch_size, num_responses, seq_len)
        candidate_attention_mask = batch['candidate_attention_mask'] # (batch_size, num_responses, seq_len)
        candidate_labels = batch['candidate_labels']          # <--- 마스킹된 레이블 사용
        mu_weights_k = batch.get('mu_weights_k', None) # (list of (batch_size,) tensors)

        batch_size, num_responses, seq_len = candidate_input_ids.shape

        # 모든 후보를 평탄화 (batch_size * num_responses, seq_len)
        all_candidate_input_ids_flat = candidate_input_ids.view(-1, seq_len)
        all_candidate_attention_mask_flat = candidate_attention_mask.view(-1, seq_len)
        all_candidate_labels_flat = candidate_labels.view(-1, seq_len) # <--- 마스킹된 레이블 평탄화

        # 정책 모델의 모든 후보에 대한 로그 확률 계산
        all_response_log_probs_policy_flat = self._get_log_probs(
            policy_model,
            all_candidate_input_ids_flat,
            all_candidate_attention_mask_flat,
            all_candidate_labels_flat  # <--- 마스킹된 레이블 전달
        )
        # (batch_size, num_responses) 형태로 복원
        all_response_log_probs_policy = all_response_log_probs_policy_flat.view(batch_size, num_responses)

        # (선택 사항) 참조 모델의 모든 후보에 대한 로그 확률 계산 (KL 정규화용)
        all_response_log_probs_ref = None
        if self.reference_model is not None:
            with torch.no_grad():
                all_response_log_probs_ref_flat = self._get_log_probs(
                    self.reference_model,
                    all_candidate_input_ids_flat,
                    all_candidate_attention_mask_flat,
                    all_candidate_labels_flat  # <--- 마스킹된 레이블 전달
                )
            # (batch_size, num_responses) 형태로 복원
            all_response_log_probs_ref = all_response_log_probs_ref_flat.view(batch_size, num_responses)

        batch_total_spo_loss = 0.0
        batch_total_kl_term = 0.0

        for i in range(batch_size): # 각 배치 샘플에 대해 반복
            sample_spo_loss = 0.0
            current_sample_log_probs_policy = all_response_log_probs_policy[i] # (num_responses,) 현재 샘플의 정책 로그 확률

            # SPO 핵심 항 계산 (Plackett-Luce 기반)
            for k in range(num_responses - 1):  # k는 0부터 n-2까지 (n-1개의 항)
                # y_tau(k): k번째 순위 응답의 원래 인덱스 및 해당 로그 확률
                chosen_for_k_log_prob = current_sample_log_probs_policy[k] # y_tau(k)의 로그 확률
                # 분모에 사용될 후보군: y_tau(k) 부터 y_tau(n-1) 까지의 응답들
                # 해당 응답들의 로그 확률
                candidates_for_denominator_k_log_probs = current_sample_log_probs_policy.gather(
                    dim=0, index=torch.tensor(list(range(k + 1, num_responses)))
                ) # (num_responses - k,)

                term_k = - (1 / self.alpha) * self._calculate_base_term(
                    chosen_for_k_log_prob,
                    candidates_for_denominator_k_log_probs
                )

                # mu_k 가중치 적용 (선택 사항)
                if mu_weights_k is not None:
                    # current_ranked_original_idx가 mu_weights_k 리스트의 유효한 인덱스인지 확인
                    if current_ranked_original_idx < len(mu_weights_k) and \
                        mu_weights_k[current_ranked_original_idx] is not None:
                        # 샘플 인덱스 i가 해당 텐서에 유효한지 확인
                        if i < mu_weights_k[current_ranked_original_idx].size(0):
                            # k번째 순위를 차지한 아이템의 '고유 가중치'를 가져옴
                            inherent_weight_of_chosen_item = mu_weights_k[current_ranked_original_idx][i]
                            term_k *= inherent_weight_of_chosen_item


                sample_spo_loss += term_k
            
            batch_total_spo_loss += sample_spo_loss

            # KL 정규화 항 계산 (최상위 순위 응답 기반)
            if self.reference_model is not None and all_response_log_probs_ref is not None:
                current_sample_log_probs_ref = all_response_log_probs_ref[i] # (num_responses,) 현재 샘플의 참조 로그 확률

                # 최상위 순위 응답 (y_tau(0))
                top_ranked_original_idx = current_sample_ranked_indices[0]
                
                top_ranked_log_prob_policy = current_sample_log_probs_policy[top_ranked_original_idx]
                top_ranked_log_prob_ref = current_sample_log_probs_ref[top_ranked_original_idx]
                
                kl_term_sample = self.beta * (top_ranked_log_prob_policy - top_ranked_log_prob_ref)
                batch_total_kl_term += kl_term_sample
        
        # 배치 전체에 대한 평균 손실
        final_spo_loss = batch_total_spo_loss / batch_size
        
        if self.reference_model is not None:
            final_kl_term = batch_total_kl_term / batch_size
            total_loss = final_spo_loss + final_kl_term
        else:
            total_loss = final_spo_loss
            
        return total_loss

# --- 2. Custom Trainer Class (compute_loss 수정) ---
class CustomSPOTrainer(Trainer):
    def __init__(self, spo_loss_fn: SPOLoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spo_loss_fn = spo_loss_fn

    # compute_loss 메서드 시그니처 수정
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # <-- num_items_in_batch 추가
        model.train()

        loss_type = inputs['type'][0] 

        if loss_type == "ranked":
            loss = self.spo_loss_fn.ranked_preference_loss(model, inputs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return (loss, None) if return_outputs else loss


# --- SPODataCollator 수정 ---
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer # 최상단에 이미 있을 수 있음
from typing import List, Dict, Any # 최상단에 이미 있을 수 있음

# --- SPODataCollator 수정 ---
class SPODataCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else: # Fallback: try to add a generic pad token
                try:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Or some other token
                    self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
                    print("Warning: Tokenizer did not have a pad_token. Added new one and set it as pad_token.")
                except:
                    raise ValueError("Tokenizer needs a pad_token or eos_token, or be configurable to add one.")
        
        if self.tokenizer.pad_token_id is None and self.tokenizer.pad_token is not None:
             self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {}

        first_item_type = features[0].get("type")
        if first_item_type is None: raise ValueError("Feature must have a 'type' key.")

        # Assuming K (num_responses_per_sample) is consistent for 'ranked' type
        # For 'best_of_n', K might be implicit or vary, handled by its specific logic.
        num_responses_per_sample = 0
        if first_item_type == "ranked":
            if not features[0].get("completions") or not isinstance(features[0]["completions"], list):
                raise ValueError("First feature of type 'ranked' must have a list of 'completions'.")
            num_responses_per_sample = len(features[0]["completions"])
            if num_responses_per_sample == 0:
                raise ValueError("Type 'ranked' features must have at least one completion.")
        elif first_item_type == "best_of_n": # For best_of_n, completions list might determine K
            num_responses_per_sample = len(features[0].get("completions", []))


        # 1. 배치 전체에서 최대 시퀀스 길이 계산 (prompt + completion)
        global_max_seq_len_in_batch = 0
        for feature in features:
            prompt_text = feature["problem"]
            responses_text = feature["completions"] # Assumed to be List[str] of length K

            # 데이터 일관성 가정: 모든 ranked feature는 K개의 completions를 가짐
            if first_item_type == "ranked" and len(responses_text) != num_responses_per_sample:
                raise ValueError(f"Data inconsistency: Feature '{feature.get('problem','N/A')}' has {len(responses_text)} completions, expected {num_responses_per_sample}.")

            for response_text in responses_text:
                full_text = prompt_text + " " + response_text # Simple concatenation
                full_tokens = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=True
                )
                global_max_seq_len_in_batch = max(global_max_seq_len_in_batch, len(full_tokens["input_ids"]))
        
        if global_max_seq_len_in_batch == 0: global_max_seq_len_in_batch = 1 # Avoid division by zero if all inputs are empty

        batch_candidate_input_ids = []
        batch_candidate_attention_mask = []
        batch_candidate_labels = []

        if first_item_type == "best_of_n":
            batch_chosen_index_in_candidates = []
            batch_mu_weights_scalar = []
        elif first_item_type == "ranked":
            batch_ranked_indices = []
            batch_mu_weights_k_list = [] # Stores List[float] for each sample

        for feature in features:
            prompt_text = feature["problem"]
            responses_text = feature["completions"] # Assumed List[str] of length K

            prompt_tokens_dict = self.tokenizer(
                prompt_text,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True # e.g., BOS + prompt_tokens
            )
            len_prompt_tokens_in_full = len(prompt_tokens_dict["input_ids"])

            sample_cand_ids, sample_cand_attn, sample_cand_labels = [], [], []

            for response_text in responses_text: # This loop runs K times
                full_text = prompt_text + " " + response_text
                full_tokens = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=True # BOS + prompt + response + EOS (or similar)
                )
                input_ids_list = full_tokens["input_ids"]
                attention_mask_list = full_tokens["attention_mask"]

                padding_len = global_max_seq_len_in_batch - len(input_ids_list)
                
                final_ids = torch.tensor(input_ids_list + [self.tokenizer.pad_token_id] * padding_len, dtype=torch.long)
                final_attn = torch.tensor(attention_mask_list + [0] * padding_len, dtype=torch.long)
                
                final_labels = final_ids.clone()
                mask_end_idx = min(len_prompt_tokens_in_full, final_labels.size(0))
                final_labels[:mask_end_idx] = -100 # Mask prompt tokens
                if padding_len > 0:
                    final_labels[-padding_len:] = -100 # Mask padding tokens
                
                sample_cand_ids.append(final_ids)
                sample_cand_attn.append(final_attn)
                sample_cand_labels.append(final_labels)
            
            # Stack K responses for this sample: (K, global_max_seq_len_in_batch)
            if not sample_cand_ids and num_responses_per_sample > 0 : # Should not happen if data is consistent
                raise ValueError(f"Feature '{prompt_text}' yielded no tokenized responses despite expecting K={num_responses_per_sample}.")
            
            # If num_responses_per_sample is 0 (e.g. for a best_of_n with no completions), stack will handle empty list if sample_cand_ids is empty
            batch_candidate_input_ids.append(torch.stack(sample_cand_ids) if sample_cand_ids else torch.empty(0, global_max_seq_len_in_batch, dtype=torch.long) )
            batch_candidate_attention_mask.append(torch.stack(sample_cand_attn) if sample_cand_attn else torch.empty(0, global_max_seq_len_in_batch, dtype=torch.long))
            batch_candidate_labels.append(torch.stack(sample_cand_labels) if sample_cand_labels else torch.empty(0, global_max_seq_len_in_batch, dtype=torch.long))


            if first_item_type == "best_of_n":
                batch_chosen_index_in_candidates.append(feature["chosen_idx"])
                batch_mu_weights_scalar.append(feature.get("mu_weight", 1.0))
            elif first_item_type == "ranked":
                batch_mu_weights_k_list.append(feature["mu_weights_k"]) # This is List[float]

        # Final batch assembly
        batch = {"type": [first_item_type] * len(features)}
        
        # Stack across batch: (B, K, global_max_seq_len_in_batch)
        # Handle case where batch_candidate_input_ids might be empty if features was empty or all num_responses_per_sample were 0
        if batch_candidate_input_ids and all(t.numel() > 0 for t in batch_candidate_input_ids if isinstance(t, torch.Tensor)): # Ensure list is not empty and tensors are not empty
            batch['candidate_input_ids'] = torch.stack(batch_candidate_input_ids)
            batch['candidate_attention_mask'] = torch.stack(batch_candidate_attention_mask)
            batch['candidate_labels'] = torch.stack(batch_candidate_labels)
        else: # Should have K elements if K > 0
            raise ValueError("No valid candidate input IDs found. Ensure features have valid completions.")

        if first_item_type == "best_of_n":
            batch['chosen_index_in_candidates'] = torch.tensor(batch_chosen_index_in_candidates, dtype=torch.long)
            batch['mu_weights'] = torch.tensor(batch_mu_weights_scalar, dtype=torch.float)
        elif first_item_type == "ranked":
            if batch_mu_weights_k_list: # List of K-element List[float]
                # Convert List[List[float]] to (B, K) tensor
                temp_mu_tensors = [torch.tensor(mu_list, dtype=torch.float) for mu_list in batch_mu_weights_k_list]
                batch['mu_weights_k'] = torch.stack(temp_mu_tensors) # (B, K)
            else: # Should have K elements if K > 0
                raise ValueError("No valid mu_weights_k found. Ensure features have valid mu_weights_k lists.")
        
        return batch
