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
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, reference_model: nn.Module = None):
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
        ranked_indices = batch['ranked_indices']              # (batch_size, num_responses)
        mu_weights_k: Optional[List[torch.Tensor]] = batch.get('mu_weights_k', None) # (list of (batch_size,) tensors)

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
            current_sample_ranked_indices = ranked_indices[i]              # (num_responses,) 현재 샘플의 순위 인덱스
            current_sample_log_probs_policy = all_response_log_probs_policy[i] # (num_responses,) 현재 샘플의 정책 로그 확률

            # SPO 핵심 항 계산 (Plackett-Luce 기반)
            for k in range(num_responses - 1):  # k는 0부터 n-2까지 (n-1개의 항)
                # y_tau(k): k번째 순위 응답의 원래 인덱스 및 해당 로그 확률
                current_ranked_original_idx = current_sample_ranked_indices[k]
                chosen_for_k_log_prob = current_sample_log_probs_policy[current_ranked_original_idx]

                # 분모에 사용될 후보군: y_tau(k) 부터 y_tau(n-1) 까지의 응답들
                # 해당 응답들의 원래 인덱스
                indices_for_denominator_k = current_sample_ranked_indices[k:]
                # 해당 응답들의 로그 확률
                candidates_for_denominator_k_log_probs = current_sample_log_probs_policy.gather(
                    dim=0, index=indices_for_denominator_k
                ) # (num_responses - k,)

                term_k = - (1 / self.alpha) * self._calculate_base_term(
                    chosen_for_k_log_prob,
                    candidates_for_denominator_k_log_probs
                )

                # mu_k 가중치 적용 (선택 사항)
                if mu_weights_k is not None and k < len(mu_weights_k):
                    # mu_weights_k[k]는 (batch_size,) 형태의 텐서여야 함
                    if mu_weights_k[k] is not None and i < mu_weights_k[k].size(0):
                         term_k *= mu_weights_k[k][i]
                    # else: # 가중치가 없거나 인덱스 오류 시 경고 (collator에서 처리됨을 가정)
                        # print(f"Warning: mu_weight for k={k}, sample={i} not found or index out of bounds.")

                sample_spo_loss += term_k
            
            batch_total_spo_loss += sample_spo_loss

            # KL 정규화 항 계산 (최상위 순위 응답 기반)
            if self.reference_model is not None and all_response_log_probs_ref is not None:
                current_sample_log_probs_ref = all_response_log_probs_ref[i] # (num_responses,) 현재 샘플의 참조 로그 확률

                # 최상위 순위 응답 (y_tau(0))
                top_ranked_original_idx = current_sample_ranked_indices[0]
                
                top_ranked_log_prob_policy = current_sample_log_probs_policy[top_ranked_original_idx]
                top_ranked_log_prob_ref = current_sample_log_probs_ref[top_ranked_original_idx]
                
                kl_term_sample = -self.beta * (top_ranked_log_prob_policy - top_ranked_log_prob_ref)
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


# --- 기존 코드의 일부 ---
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Dict, Any

# --- 기존 코드의 일부 ---
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List, Dict, Any

# --- SPODataCollator 수정 ---
class SPODataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None: # Ensure pad token is set
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        first_item_type = features[0]["type"]
        for feature in features:
            if feature["type"] != first_item_type:
                raise ValueError("All items in a batch must be of the same preference type (best_of_n or ranked).")

        batch = {
            "type": [first_item_type] * len(features)
        }

        batch_candidate_input_ids = []
        batch_candidate_attention_mask = []
        batch_candidate_labels = [] # <--- 수정: 마스킹된 레이블을 저장할 리스트 추가

        if first_item_type == "best_of_n":
            batch_chosen_index_in_candidates = []
            batch_mu_weights = []
        elif first_item_type == "ranked":
            batch_ranked_indices = []
            batch_mu_weights_k_list = []

        max_seq_len_in_batch = 0 # 배치 내 최대 시퀀스 길이 동적 결정

        for feature in features:
            prompt_text = feature["prompt"]
            responses_text = feature["responses"]

            # 프롬프트 텍스트를 먼저 토큰화하여 실제 프롬프트의 토큰 길이를 얻습니다.
            # 이는 full_text에서 프롬프트 부분을 식별하는 데 사용됩니다.
            # add_special_tokens=True로 설정하여 문장 시작에 추가될 수 있는 BOS 토큰 등을 포함시킵니다.
            # 이 길이는 full_text에서 response가 시작되기 직전까지의 길이를 나타냅니다.
            # (주의: tokenizer나 모델에 따라 프롬프트와 응답 결합 방식이 다르면 이 부분이 더 정교해져야 할 수 있습니다)
            prompt_only_ids = self.tokenizer.encode(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length
            )
            len_prompt_tokens_in_full = len(prompt_only_ids)


            sample_candidate_input_ids = []
            sample_candidate_attention_mask = []
            sample_candidate_labels = [] # <--- 수정: 현재 샘플의 마스킹된 레이블 리스트

            for response_text in responses_text:
                full_text = prompt_text + response_text
                full_tokens = self.tokenizer(
                    full_text,
                    # return_tensors="pt", # 여기서는 바로 tensor로 만들 필요 없음
                    max_length=self.max_length,
                    truncation=True,
                    # add_special_tokens=True is default and desired
                )
                current_input_ids = torch.tensor(full_tokens.input_ids)
                current_attention_mask = torch.tensor(full_tokens.attention_mask)

                # 레이블 생성: input_ids를 복사한 후 프롬프트 부분을 -100으로 마스킹
                current_labels = current_input_ids.clone()
                
                # 프롬프트 부분 마스킹
                # 실제 full_tokens.input_ids 와 prompt_only_ids를 비교하여 정확한 마스킹 길이를 결정하는 것이 더 안전할 수 있으나,
                # 일반적인 경우 (tokenizer가 prompt와 prompt+response를 일관되게 처리할 때)
                # 미리 계산된 len_prompt_tokens_in_full을 사용할 수 있습니다.
                # 만약 full_tokens.input_ids 시작 부분이 prompt_only_ids와 정확히 일치하지 않는 경우가 있다면
                # (예: 프롬프트 끝과 응답 시작 부분의 특수 토큰 처리 방식 차이) 주의가 필요합니다.
                mask_end_idx = min(len_prompt_tokens_in_full, current_labels.size(0))
                current_labels[:mask_end_idx] = -100

                sample_candidate_input_ids.append(current_input_ids)
                sample_candidate_attention_mask.append(current_attention_mask)
                sample_candidate_labels.append(current_labels) # <--- 수정: 마스킹된 레이블 추가
                
                if current_input_ids.size(0) > max_seq_len_in_batch:
                    max_seq_len_in_batch = current_input_ids.size(0)

            # 현재 샘플 내 후보들을 max_seq_len_in_batch에 맞춰 패딩
            padded_ids_for_sample = []
            padded_attn_for_sample = []
            padded_labels_for_sample = [] # <--- 수정: 패딩된 레이블을 저장할 리스트

            for ids_t, attn_t, labels_t in zip(sample_candidate_input_ids, sample_candidate_attention_mask, sample_candidate_labels):
                padding_len = max_seq_len_in_batch - ids_t.size(0)
                padded_ids_for_sample.append(F.pad(ids_t, (0, padding_len), value=self.tokenizer.pad_token_id))
                padded_attn_for_sample.append(F.pad(attn_t, (0, padding_len), value=0))
                padded_labels_for_sample.append(F.pad(labels_t, (0, padding_len), value=-100)) # <--- 수정: 레이블은 -100으로 패딩

            batch_candidate_input_ids.append(torch.stack(padded_ids_for_sample))
            batch_candidate_attention_mask.append(torch.stack(padded_attn_for_sample))
            batch_candidate_labels.append(torch.stack(padded_labels_for_sample)) # <--- 수정: 배치에 마스킹 및 패딩된 레이블 추가

            if first_item_type == "best_of_n":
                batch_chosen_index_in_candidates.append(feature["chosen_idx"])
                batch_mu_weights.append(feature.get("mu_weight", 1.0)) # get으로 기본값 처리
            elif first_item_type == "ranked":
                batch_ranked_indices.append(feature["ranked_indices"])
                batch_mu_weights_k_list.append(feature.get("mu_weights_k", [])) # get으로 기본값 처리

        batch['candidate_input_ids'] = torch.stack(batch_candidate_input_ids)
        batch['candidate_attention_mask'] = torch.stack(batch_candidate_attention_mask)
        batch['candidate_labels'] = torch.stack(batch_candidate_labels) # <--- 수정: 최종 배치 딕셔너리에 추가

        if first_item_type == "best_of_n":
            batch['chosen_index_in_candidates'] = torch.tensor(batch_chosen_index_in_candidates, dtype=torch.long)
            batch['mu_weights'] = torch.tensor(batch_mu_weights, dtype=torch.float)
        elif first_item_type == "ranked":
            batch['ranked_indices'] = torch.tensor(batch_ranked_indices, dtype=torch.long)
            if batch_mu_weights_k_list and any(batch_mu_weights_k_list): # 비어있지 않고, 내부 리스트도 내용이 있을 때
                 # 모든 샘플의 mu_weights_k 리스트 길이가 동일하다고 가정
                if all(len(lst) == len(batch_mu_weights_k_list[0]) for lst in batch_mu_weights_k_list if lst) or not any(len(lst) != len(batch_mu_weights_k_list[0]) for lst in batch_mu_weights_k_list if lst and batch_mu_weights_k_list[0]):
                    # Check if there are actual values to transpose
                    if batch_mu_weights_k_list[0]: # Check if the first list is not empty
                        transposed_mu_k = list(map(list, zip(*[w_list if w_list else [0.0]*len(batch_mu_weights_k_list[0]) for w_list in batch_mu_weights_k_list]))) # Handle empty lists with zeros
                        batch['mu_weights_k'] = [torch.tensor(m, dtype=torch.float) for m in transposed_mu_k]
                    else: # All lists are empty
                        batch['mu_weights_k'] = []
                else: # 가변 길이 mu_weights_k는 현재 방식으로는 처리 어려움. 경고 또는 오류 처리 필요.
                    # 여기서는 단순화를 위해 빈 리스트로 처리하거나, 예외 발생시킬 수 있음
                    print("Warning: mu_weights_k lists have variable lengths. This is not fully supported by current batching.")
                    batch['mu_weights_k'] = [] # 혹은 적절한 오류 처리
            else: # mu_weights_k가 없거나 모두 비어있는 경우
                batch['mu_weights_k'] = []
        return batch

# filepath: spo.py
import torch
from transformers import AutoTokenizer # 필요한 경우 AutoTokenizer를 임포트합니다.

# --- RankedSPODataCollator 수정 ---
class RankedSPODataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None: # Ensure pad token is set
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # If no eos_token either, raise an error as a pad token is essential for padding
                raise ValueError(
                    "The tokenizer does not have a pad_token. "
                    "Please set model.config.pad_token_id or tokenizer.pad_token directly. "
                    "If an eos_token is available, it can be used as a pad_token."
                )
        # Ensure pad_token_id is also set if pad_token was set from eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.pad_token is not None:
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)


    def __call__(self, features: list) -> dict:
        if not features:
            return {}

        first_item_type = features[0].get("type")
        if first_item_type is None:
            raise ValueError("The first feature must have a 'type' key.")

        for feature in features:
            feature_type = feature.get("type")
            if feature_type is None:
                raise ValueError("Each feature must have a 'type' key.")
            if feature_type != first_item_type:
                raise ValueError("All items in a batch must be of the same preference type (best_of_n or ranked).")

        batch: Dict[str, Union[List[Any], torch.Tensor, Any]] = {
            "type": [first_item_type] * len(features)
        }

        batch_candidate_input_ids = []
        batch_candidate_attention_mask = []
        batch_candidate_label_ids = [] # 마스킹된 레이블을 저장할 리스트

        if first_item_type == "best_of_n":
            batch_chosen_index_in_candidates = []
            batch_mu_weights = []
        elif first_item_type == "ranked":
            batch_ranked_indices = []
            batch_mu_weights_k_list = []

        # Precompute the maximum sequence length in the batch after truncation
        max_seq_len_in_batch = 0
        for feature in features:
            prompt_text = feature["prompt"]
            responses_text = feature["responses"]
            for response_text in responses_text:
                full_text = prompt_text + response_text
                full_tokens = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=True
                ) # type: ignore
                max_seq_len_in_batch = max(max_seq_len_in_batch, len(full_tokens.input_ids))
        
        # If max_seq_len_in_batch is 0 (e.g., all inputs were empty), default to a minimal length like 1
        # or handle as an error, depending on desired behavior. Here, we ensure it's at least 1.
        if max_seq_len_in_batch == 0:
             max_seq_len_in_batch = 1


        for feature in features:
            prompt_text = feature["prompt"]
            responses_text = feature["responses"]

            prompt_only_ids = self.tokenizer.encode(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length
            )
            len_prompt_tokens_in_full = len(prompt_only_ids)

            sample_candidate_input_ids = []
            sample_candidate_attention_mask = []
            sample_candidate_label_ids = [] # 현재 샘플의 마스킹된 레이블 리스트

            for response_text in responses_text:
                full_text = prompt_text + response_text
                full_tokens = self.tokenizer(
                    full_text,
                    max_length=self.max_length, # Use the pre-calculated max_seq_len_in_batch for padding consistency
                    truncation=True,
                    add_special_tokens=True,
                    # padding="max_length", # Padding will be handled manually after collecting all items for the sample
                ) # type: ignore
                current_input_ids = torch.tensor(full_tokens.input_ids)
                current_attention_mask = torch.tensor(full_tokens.attention_mask)

                current_labels = current_input_ids.clone()
                mask_end_idx = min(len_prompt_tokens_in_full, current_labels.size(0))
                current_labels[:mask_end_idx] = -100

                sample_candidate_input_ids.append(current_input_ids)
                sample_candidate_attention_mask.append(current_attention_mask)
                sample_candidate_label_ids.append(current_labels)
            
            # Pad all candidates for the current sample to max_seq_len_in_batch
            padded_ids_for_sample_list = []
            padded_attn_for_sample_list = []
            padded_labels_for_sample_list = []

            for ids_t, attn_t, labels_t in zip(sample_candidate_input_ids, sample_candidate_attention_mask, sample_candidate_label_ids):
                seq_len = ids_t.size(0)
                
                padding_needed = max_seq_len_in_batch - seq_len
                
                padded_ids = torch.cat([ids_t, torch.full((padding_needed,), self.tokenizer.pad_token_id, dtype=torch.long)], dim=0)
                padded_attn = torch.cat([attn_t, torch.zeros(padding_needed, dtype=torch.long)], dim=0)
                padded_labels = torch.cat([labels_t, torch.full((padding_needed,), -100, dtype=torch.long)], dim=0)

                padded_ids_for_sample_list.append(padded_ids)
                padded_attn_for_sample_list.append(padded_attn)
                padded_labels_for_sample_list.append(padded_labels)

            if not padded_ids_for_sample_list: # Should not happen if responses_text is not empty
                # Handle empty sample candidates if necessary, e.g., by creating dummy tensors
                # For now, assume responses_text always yields some candidates
                # If it can be empty, one might need to add placeholder tensors of appropriate shape.
                # Example:
                # num_candidates_expected = 1 # Or some other default
                # padded_ids_for_sample = torch.full((num_candidates_expected, max_seq_len_in_batch), fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
                # padded_attn_for_sample = torch.zeros((num_candidates_expected, max_seq_len_in_batch), dtype=torch.long)
                # padded_labels_for_sample = torch.full((num_candidates_expected, max_seq_len_in_batch), fill_value=-100, dtype=torch.long)
                # Fallback or error for empty candidates
                if not responses_text : # if responses_text was empty for this feature
                    # Create a dummy tensor to maintain batch structure if required
                    # This assumes at least one "candidate" slot, filled with padding.
                    # Adjust num_dummy_candidates as per requirements.
                    num_dummy_candidates = 1 
                    padded_ids_for_sample = torch.full((num_dummy_candidates, max_seq_len_in_batch), self.tokenizer.pad_token_id, dtype=torch.long)
                    padded_attn_for_sample = torch.zeros((num_dummy_candidates, max_seq_len_in_batch), dtype=torch.long)
                    padded_labels_for_sample = torch.full((num_dummy_candidates, max_seq_len_in_batch), -100, dtype=torch.long)
                else: # This case should ideally not be reached if responses_text is processed
                    raise ValueError("padded_ids_for_sample_list is empty unexpectedly.")

            else:
                padded_ids_for_sample = torch.stack(padded_ids_for_sample_list)
                padded_attn_for_sample = torch.stack(padded_attn_for_sample_list)
                padded_labels_for_sample = torch.stack(padded_labels_for_sample_list)


            batch_candidate_input_ids.append(padded_ids_for_sample)
            batch_candidate_attention_mask.append(padded_attn_for_sample)
            batch_candidate_label_ids.append(padded_labels_for_sample)

            if first_item_type == "ranked":
                batch_ranked_indices.append(feature["ranked_indices"])
                batch_mu_weights_k_list.append(feature.get("mu_weights_k", []))

        batch['candidate_input_ids'] = torch.stack(batch_candidate_input_ids)
        batch['candidate_attention_mask'] = torch.stack(batch_candidate_attention_mask)
        batch['candidate_label_ids'] = torch.stack(batch_candidate_label_ids)

        if first_item_type == "ranked":
            batch['ranked_indices'] = batch_ranked_indices # List of lists of integers

            if not batch_mu_weights_k_list:
                batch['mu_weights_k'] = []
            else:
                non_empty_lists = [lst for lst in batch_mu_weights_k_list if lst]
                if not non_empty_lists:
                    batch['mu_weights_k'] = []
                else:
                    first_len = len(non_empty_lists[0])
                    if not all(len(lst) == first_len for lst in non_empty_lists):
                        # Option 1: Raise error (as in original intent)
                        raise ValueError("mu_weights_k lists have variable lengths among non-empty lists, which is not supported.")
                        # Option 2: Print warning and pad/truncate (more complex, not implemented here)
                        # print("Warning: mu_weights_k lists have variable lengths. This may lead to issues.")
                        # batch['mu_weights_k'] = [] # Or handle by padding/truncating
                    
                    # Pad empty lists to first_len before transposing
                    processed_mu_weights_k_list = []
                    for w_list in batch_mu_weights_k_list:
                        if w_list:
                            processed_mu_weights_k_list.append(w_list)
                        else:
                            processed_mu_weights_k_list.append([0.0] * first_len) # Pad empty original lists

                    if not processed_mu_weights_k_list or not processed_mu_weights_k_list[0]: # Should be non-empty if non_empty_lists was populated
                        batch['mu_weights_k'] = []
                    else:
                        transposed_mu_k = list(map(list, zip(*processed_mu_weights_k_list)))
                        batch['mu_weights_k'] = [torch.tensor(m, dtype=torch.float) for m in transposed_mu_k]
        return batch