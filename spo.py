import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Any, Optional
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn as nn
import torch.nn.functional as F # _get_log_probs 내에서 F.log_softmax 사용 가정
from typing import Dict, List # Dict는 batch 타입 어노테이션, List는 mu_weights_k 타입 어노테이션에 사용

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class SPOLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.01,
                 beta: float = 0.1,
                 gamma_score: float = 0.01,
                 eta_decay: float = 1.0,
                 mu_scale_factor: float = 1.0,
                 reference_model: Optional[nn.Module] = None,
                 use_global_kl: bool = False
                ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma_score = gamma_score
        self.eta_decay = eta_decay
        self.mu_scale_factor = mu_scale_factor
        self.use_global_kl = use_global_kl

        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval()

    def _get_sequence_log_probs_from_logits(self,
                                            logits: torch.Tensor,
                                            labels: torch.Tensor
                                           ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        log_probs_all_tokens = F.log_softmax(shift_logits, dim=-1)
        loss_fct = nn.NLLLoss(ignore_index=-100, reduction='none')
        per_token_neg_log_likelihood = loss_fct(
            log_probs_all_tokens.view(-1, log_probs_all_tokens.size(-1)),
            shift_labels.view(-1)
        )
        per_token_neg_log_likelihood = per_token_neg_log_likelihood.view(labels.size(0), -1)
        sequence_log_probs = -per_token_neg_log_likelihood.sum(dim=-1)
        return sequence_log_probs

    def _calculate_preference_term(self, chosen_log_prob: torch.Tensor, candidates_log_probs: torch.Tensor) -> torch.Tensor:
        chosen_term = chosen_log_prob * self.alpha
        sum_candidates_term = torch.logsumexp(candidates_log_probs * self.alpha, dim=-1)
        return chosen_term - sum_candidates_term

    def _calculate_mu_k(self,
                        current_k: int,
                        scores_for_current_sample_ranked: torch.Tensor, # 1D 텐서
                        batch_avg_V_term: torch.Tensor
                       ) -> torch.Tensor:
        # scores_for_current_sample_ranked가 비어있거나 current_k가 범위를 벗어나는 경우는
        # 호출하는 쪽(forward)에서 num_responses 검사를 통해 사전에 방지한다고 가정.
        scores_in_C_k = scores_for_current_sample_ranked[current_k:]
        
        # scores_in_C_k가 비어있는 경우는 (current_k == num_responses 일 때), sum_scores_in_C_k는 0이 됨.
        # 이 경우 V_ik_current도 0이 될 수 있음 (0^gamma_score). 이는 의도된 동작일 수 있음.
        sum_scores_in_C_k = torch.sum(scores_in_C_k)
        V_ik_current = sum_scores_in_C_k**self.gamma_score
        
        quality_arg = V_ik_current - batch_avg_V_term
        overall_quality_weight = self.mu_scale_factor * torch.sigmoid(quality_arg)
        
        final_mu_k = (self.eta_decay**current_k) * overall_quality_weight
        return final_mu_k

    def _calculate_tokenwise_kl_on_sequences(self,
                                             policy_logits: torch.Tensor,
                                             input_ids: torch.Tensor,
                                             attention_mask: torch.Tensor,
                                             labels: torch.Tensor
                                            ) -> torch.Tensor:
        # self.reference_model이 None인 경우는 forward에서 beta > 0일 때 미리 체크함.
        with torch.no_grad():
            ref_outputs = self.reference_model(input_ids=input_ids, attention_mask=attention_mask) # type: ignore
            ref_logits = ref_outputs.logits

        ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        policy_log_probs = F.log_softmax(policy_logits[:, :-1, :], dim=-1)
        
        shifted_labels = labels[:, 1:].contiguous()
        valid_token_mask = (shifted_labels != -100)

        kl_div_per_token_position = F.kl_div(
            input=ref_log_probs, target=policy_log_probs, reduction='none', log_target=True
        ).sum(dim=-1)
        
        masked_kl_div = kl_div_per_token_position * valid_token_mask
        sequence_kl_sum = masked_kl_div.sum(dim=-1)
        
        num_valid_tokens_per_sequence = valid_token_mask.sum(dim=-1).float()
        # 0으로 나누는 것 방지: num_valid_tokens_per_sequence가 0이면 KL도 0이 되도록 함
        # (masked_kl_div.sum도 0이므로 결과적으로 0/0 -> nan 대신 0이 됨)
        avg_kl_per_sequence = torch.where(
            num_valid_tokens_per_sequence > 0,
            sequence_kl_sum / (num_valid_tokens_per_sequence + 1e-8), # 작은 값 더해서 0으로 나누기 방지
            torch.zeros_like(sequence_kl_sum)
        )
        return avg_kl_per_sequence.mean() if avg_kl_per_sequence.nelement() > 0 else torch.tensor(0.0, device=policy_logits.device)


    def forward(self,
                policy_model: nn.Module,
                batch: Dict[str, torch.Tensor]
               ) -> torch.Tensor:
        
        candidate_input_ids = batch.get('candidate_input_ids')
        candidate_attention_mask = batch.get('candidate_attention_mask')
        candidate_labels = batch.get('candidate_labels')
        external_scores_ranked = batch.get('prm_avg_scores')

        if any(t is None for t in [candidate_input_ids, candidate_attention_mask, candidate_labels, external_scores_ranked]):
            raise KeyError("One or more required keys ('candidate_input_ids', 'candidate_attention_mask', "
                           "'candidate_labels', 'external_scores_ranked') are missing from the batch.")

        batch_size, num_responses, seq_len = candidate_input_ids.shape
        device = candidate_input_ids.device

        if num_responses == 0: # 응답이 없는 경우 손실 0 반환 또는 에러
            return torch.tensor(0.0, device=device, requires_grad=True) # 학습 가능하도록

        flat_input_ids = candidate_input_ids.reshape(-1, seq_len)
        flat_attention_mask = candidate_attention_mask.reshape(-1, seq_len)
        flat_labels = candidate_labels.reshape(-1, seq_len)

        policy_outputs = policy_model(input_ids=flat_input_ids, attention_mask=flat_attention_mask, labels=flat_labels)
        policy_logits_flat = policy_outputs.logits
        
        all_response_log_probs_policy_flat = self._get_sequence_log_probs_from_logits(
            policy_logits_flat, flat_labels
        )
        all_response_log_probs_policy = all_response_log_probs_policy_flat.view(batch_size, num_responses)

        # mu_k 계산을 위한 batch_avg_V_term 계산
        batch_avg_V_term = torch.tensor(0.0, device=device)
        if num_responses > 0 : # V_term 계산은 응답이 있을 때만 의미 있음
            all_V_values_in_batch_for_mu = []
            for i in range(batch_size):
                current_sample_scores_ranked = external_scores_ranked[i]
                for k_loop_prep in range(num_responses):
                    scores_in_C_k_prep = current_sample_scores_ranked[k_loop_prep:]
                    # scores_in_C_k_prep가 비어있지 않음을 보장 (k_loop_prep < num_responses 이므로)
                    sum_scores_in_C_k_prep = torch.sum(scores_in_C_k_prep)
                    V_ik_prep = sum_scores_in_C_k_prep**self.gamma_score
                    all_V_values_in_batch_for_mu.append(V_ik_prep)
            
            if len(all_V_values_in_batch_for_mu) > 0:
                batch_avg_V_term = torch.mean(torch.stack(all_V_values_in_batch_for_mu))
        
        # 선호도 손실 계산
        total_preference_loss_terms = []
        if num_responses > 1: # 비교할 쌍이 있는 경우에만 계산
            for i in range(batch_size):
                current_sample_log_probs_policy_ranked = all_response_log_probs_policy[i]
                current_sample_scores_ranked = external_scores_ranked[i]

                for k in range(num_responses - 1):
                    chosen_log_prob = current_sample_log_probs_policy_ranked[k]
                    candidates_denominator_log_probs = current_sample_log_probs_policy_ranked[k:]
                    
                    term_k_log_ratio = self._calculate_preference_term(
                        chosen_log_prob,
                        candidates_denominator_log_probs
                    )
                    mu_k = self._calculate_mu_k(
                        k, current_sample_scores_ranked, batch_avg_V_term
                    )
                    term_k = -(1.0 / self.alpha) * term_k_log_ratio * mu_k
                    total_preference_loss_terms.append(term_k)
        
        if len(total_preference_loss_terms) > 0:
            # 각 term_k는 스칼라이므로, stack 후 mean 또는 sum / count
            final_preference_loss = torch.mean(torch.stack(total_preference_loss_terms))
        else: # num_responses <= 1 이거나 batch_size = 0
            final_preference_loss = torch.tensor(0.0, device=device)
        
        # KL 발산 손실 계산
        kl_divergence_loss = torch.tensor(0.0, device=device)
        if self.beta > 0:
            if self.reference_model is None:
                raise ValueError("reference_model is required for KL divergence calculation when beta > 0.")

            kl_input_ids, kl_attn_mask, kl_labels, kl_policy_logits = None, None, None, None
            kl_data_available = False

            if self.use_global_kl:
                if not all(key in batch for key in ['online_input_ids', 'online_attention_mask', 'online_labels']):
                    raise KeyError("Global KL divergence requires 'online_input_ids', 'online_attention_mask', "
                                   "and 'online_labels' in batch when use_global_kl is True.")
                
                online_input_ids = batch['online_input_ids']
                if online_input_ids.nelement() > 0:
                    online_policy_outputs = policy_model(
                        input_ids=online_input_ids,
                        attention_mask=batch['online_attention_mask'],
                        labels=batch['online_labels']
                    )
                    kl_input_ids, kl_attn_mask, kl_labels, kl_policy_logits = \
                        online_input_ids, batch['online_attention_mask'], batch['online_labels'], online_policy_outputs.logits
                    kl_data_available = True
            else: # 데이터셋 내 KL
                if flat_input_ids.nelement() > 0:
                    kl_input_ids, kl_attn_mask, kl_labels, kl_policy_logits = \
                        flat_input_ids, flat_attention_mask, flat_labels, policy_logits_flat
                    kl_data_available = True
            
            if kl_data_available:
                kl_divergence_loss = self._calculate_tokenwise_kl_on_sequences(
                    kl_policy_logits, kl_input_ids, kl_attn_mask, kl_labels
                )
        
        total_loss = final_preference_loss + self.beta * kl_divergence_loss
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
            loss = self.spo_loss_fn.forward(model, inputs)
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
    def __init__(self, tokenizer,instruction, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction

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
                full_text = self.instruction +prompt_text + " " + response_text # Simple concatenation
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
                batch_mu_weights_k_list.append(feature["prm_avg_scores"]) # This is List[float]

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
            batch['prm_avg_scores'] = torch.tensor(batch_mu_weights_scalar, dtype=torch.float)
        elif first_item_type == "ranked":
            if batch_mu_weights_k_list: # List of K-element List[float]
                # Convert List[List[float]] to (B, K) tensor
                temp_mu_tensors = [torch.tensor(mu_list, dtype=torch.float) for mu_list in batch_mu_weights_k_list]
                batch['prm_avg_scores'] = torch.stack(temp_mu_tensors) # (B, K)
            else: # Should have K elements if K > 0
                raise ValueError("No valid mu_weights_k found. Ensure features have valid mu_weights_k lists.")
        
        return batch
