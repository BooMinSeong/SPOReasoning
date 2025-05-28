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
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer # 최상단에 이미 있을 수 있음
from typing import List, Dict, Any # 최상단에 이미 있을 수 있음

# --- SPODataCollator 수정 ---
class SPODataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None: # Ensure pad token is set
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(
                    "The tokenizer does not have a pad_token or an eos_token to use as a pad_token. "
                    "Please set model.config.pad_token_id or tokenizer.pad_token directly."
                )
        # Ensure pad_token_id is also set if pad_token was set from eos_token or manually
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
                raise ValueError("All items in a batch must be of the same preference type (e.g., ranked).")

        batch = {
            "type": [first_item_type] * len(features)
        }

        # --- 수정된 부분 시작 ---
        # 1. 배치 전체에서 최대 시퀀스 길이 계산
        global_max_seq_len_in_batch = 0
        for feature in features:
            prompt_text = feature["problem"]
            responses_text = feature["completions"]
            for response_text in responses_text:
                full_text = prompt_text + response_text
                # 토큰화하여 실제 길이 확인 (패딩/자르기 전 길이 아님, max_length 적용 후 길이)
                full_tokens = self.tokenizer(
                    full_text,
                    max_length=self.max_length, # HuggingFace tokenizer의 max_length
                    truncation=True,
                    add_special_tokens=True
                )
                global_max_seq_len_in_batch = max(global_max_seq_len_in_batch, len(full_tokens.input_ids))
        
        # 모든 입력이 비어있는 등의 극단적인 경우 처리
        if global_max_seq_len_in_batch == 0:
            global_max_seq_len_in_batch = 1 
        # --- 수정된 부분 끝 ---

        batch_candidate_input_ids = []
        batch_candidate_attention_mask = []
        batch_candidate_labels = []

        if first_item_type == "best_of_n": # 이 부분은 현재 예제 데이터에는 없지만, 완전성을 위해 남겨둡니다.
            batch_chosen_index_in_candidates = []
            batch_mu_weights = []
        elif first_item_type == "ranked":
            batch_ranked_indices = []
            batch_mu_weights_k_list = [] # 각 샘플의 mu_weights_k 리스트를 담을 리스트

        for feature in features:
            prompt_text = feature["problem"]
            responses_text = feature["completions"]

            prompt_only_ids = self.tokenizer.encode(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length # 프롬프트 자체도 너무 길면 잘라냄
            )
            len_prompt_tokens_in_full = len(prompt_only_ids)

            sample_candidate_input_ids = []
            sample_candidate_attention_mask = []
            sample_candidate_labels = []

            for response_text in responses_text:
                full_text = prompt_text + response_text
                full_tokens = self.tokenizer(
                    full_text,
                    max_length=self.max_length, # 여기서 max_length는 tokenizer의 최대 허용 길이
                    truncation=True,
                    add_special_tokens=True,
                    # padding은 수동으로 하므로 여기서는 False 또는 지정 안 함
                )
                current_input_ids = torch.tensor(full_tokens.input_ids)
                current_attention_mask = torch.tensor(full_tokens.attention_mask)

                current_labels = current_input_ids.clone()
                # 프롬프트 부분 마스킹 시, 실제 토큰화된 프롬프트 길이를 사용해야 함
                # self.tokenizer(prompt_text, add_special_tokens=True) 로 얻은 길이를 사용
                # 혹은 full_text에서 prompt_text가 끝나는 지점을 찾아야 함
                # 여기서는 len_prompt_tokens_in_full 사용 (BOS 토큰 등 고려)
                mask_end_idx = min(len_prompt_tokens_in_full, current_labels.size(0))
                current_labels[:mask_end_idx] = -100

                sample_candidate_input_ids.append(current_input_ids)
                sample_candidate_attention_mask.append(current_attention_mask)
                sample_candidate_labels.append(current_labels)

            # 현재 샘플 내 후보들을 global_max_seq_len_in_batch에 맞춰 패딩
            padded_ids_for_sample_list = []
            padded_attn_for_sample_list = []
            padded_labels_for_sample_list = []

            if not sample_candidate_input_ids: # feature에 response가 하나도 없는 경우
                # 이런 경우, ranked_indices 등 다른 필드도 비어있거나 길이가 0이어야 함.
                # dummy 텐서를 만들거나 에러를 발생시킬 수 있음.
                # 현재 데이터 구조상 responses가 항상 존재한다고 가정.
                # 만약 responses가 비어있을 수 있다면, 이 부분을 더 견고하게 처리해야 함.
                # 예: num_expected_responses = len(features[0]["responses"]) # 첫번째 아이템 기준으로
                #     dummy_ids = torch.full((num_expected_responses, global_max_seq_len_in_batch), self.tokenizer.pad_token_id, dtype=torch.long)
                #     dummy_attn = torch.zeros((num_expected_responses, global_max_seq_len_in_batch), dtype=torch.long)
                #     dummy_labels = torch.full((num_expected_responses, global_max_seq_len_in_batch), -100, dtype=torch.long)
                #     batch_candidate_input_ids.append(dummy_ids)
                #     ...
                #     continue # 다음 feature로
                 pass # 아래 stack에서 오류가 날 수 있으므로, 빈 리스트를 stack하지 않도록 주의

            for ids_t, attn_t, labels_t in zip(sample_candidate_input_ids, sample_candidate_attention_mask, sample_candidate_labels):
                padding_len = global_max_seq_len_in_batch - ids_t.size(0)
                if padding_len < 0: # global_max_seq_len_in_batch 보다 긴 시퀀스가 있는 경우 (이론상 발생 안해야 함)
                    padding_len = 0 # 자르기는 이미 tokenizer에서 수행됨

                padded_ids_for_sample_list.append(F.pad(ids_t, (0, padding_len), value=self.tokenizer.pad_token_id))
                padded_attn_for_sample_list.append(F.pad(attn_t, (0, padding_len), value=0))
                padded_labels_for_sample_list.append(F.pad(labels_t, (0, padding_len), value=-100))

            if padded_ids_for_sample_list: # 실제 패딩된 결과가 있을 때만 스택
                batch_candidate_input_ids.append(torch.stack(padded_ids_for_sample_list))
                batch_candidate_attention_mask.append(torch.stack(padded_attn_for_sample_list))
                batch_candidate_labels.append(torch.stack(padded_labels_for_sample_list))
            elif responses_text : # responses_text는 있었는데, 패딩 리스트가 빈 경우 (로직 오류 가능성)
                 # 이 경우를 대비해 더미 텐서 또는 오류 처리 필요
                 # 현재 예제 데이터에서는 responses가 항상 4개이므로 이 분기는 잘 타지 않음
                 # 만약 responses가 비어있을 수 있다면, 여기서 빈 텐서를 추가하거나 해야 함.
                 # 예: num_responses = len(feature["responses"]) # 또는 ranked_indices 길이
                 #    dummy_shape = (num_responses if num_responses > 0 else 1, global_max_seq_len_in_batch)
                 #    batch_candidate_input_ids.append(torch.full(dummy_shape, self.tokenizer.pad_token_id, dtype=torch.long))
                 #    ...
                 print(f"Warning: Feature with prompt '{prompt_text}' had responses but resulted in empty padded lists.")


            if first_item_type == "best_of_n":
                batch_chosen_index_in_candidates.append(feature["chosen_idx"])
                batch_mu_weights.append(feature.get("mu_weight", 1.0))
            elif first_item_type == "ranked":
                batch_ranked_indices.append(torch.tensor(feature["ranked_indices"], dtype=torch.long)) # 리스트를 바로 넣고 나중에 처리하거나, 여기서 텐서화
                batch_mu_weights_k_list.append(feature.get("mu_weights_k", []))


        # 모든 샘플에 대한 처리가 끝난 후, 최종적으로 배치 텐서들을 만듭니다.
        # batch_candidate_input_ids 리스트 내의 모든 텐서들은 이제 동일한 shape[1] (global_max_seq_len_in_batch)을 가져야 합니다.
        if not batch_candidate_input_ids: # 만약 전체 배치가 비어있거나, 처리 후 아무것도 남지 않았다면
            # 빈 딕셔너리 또는 적절한 오류 처리
            # 이럴 경우 Trainer에서 오류 발생 가능성 높음
            if features: # 원본 features는 있었는데 결과가 없다면 문제
                 raise ValueError("Data collator processed features but resulted in an empty batch for candidate tensors.")
            return {} # 원본 features 자체가 비었다면 빈 딕셔너리 반환은 합리적

        batch['candidate_input_ids'] = torch.stack(batch_candidate_input_ids)
        batch['candidate_attention_mask'] = torch.stack(batch_candidate_attention_mask)
        batch['candidate_labels'] = torch.stack(batch_candidate_labels)

        if first_item_type == "best_of_n":
            batch['chosen_index_in_candidates'] = torch.tensor(batch_chosen_index_in_candidates, dtype=torch.long)
            batch['mu_weights'] = torch.tensor(batch_mu_weights, dtype=torch.float)
        elif first_item_type == "ranked":
            # batch_ranked_indices는 이미 텐서의 리스트일 수 있으므로, 필요시 torch.stack 사용
            # 현재 로직에서는 각 feature의 ranked_indices를 tensor로 변환 후 리스트에 추가. 이를 stack.
            batch['ranked_indices'] = torch.stack(batch_ranked_indices) # 각 요소가 (num_responses,) 형태의 텐서이므로 stack하면 (batch_size, num_responses)

            if batch_mu_weights_k_list and any(batch_mu_weights_k_list):
                # 모든 샘플의 mu_weights_k 리스트 길이가 동일한지 확인 (또는 가장 긴 길이에 맞춰 패딩)
                # 현재 데이터는 길이가 3으로 동일
                try:
                    # Check if all inner lists have the same length if they are not empty
                    non_empty_lengths = [len(lst) for lst in batch_mu_weights_k_list if lst]
                    if not non_empty_lengths: # all lists are empty
                        batch['mu_weights_k'] = []
                    elif not all(l == non_empty_lengths[0] for l in non_empty_lengths):
                        # 가변 길이 처리: 가장 긴 길이에 맞춰 0.0 등으로 패딩하거나 오류 발생
                        # 여기서는 동일하다고 가정하고 진행 (예제 데이터는 동일)
                        # 혹은, 패딩 로직 추가
                        print("Warning: mu_weights_k lists have variable non-empty lengths. This might lead to errors or require padding.")
                        # Fallback: 가장 흔한 길이 또는 첫번째 요소의 길이로 통일 시도 (위험할 수 있음)
                        # max_len_mu = max(non_empty_lengths) if non_empty_lengths else 0
                        # padded_mu_weights_k_list = []
                        # for w_list in batch_mu_weights_k_list:
                        #    padded_mu_weights_k_list.append(w_list + [0.0] * (max_len_mu - len(w_list)))
                        # transposed_mu_k = list(map(list, zip(*padded_mu_weights_k_list)))
                        # batch['mu_weights_k'] = [torch.tensor(m, dtype=torch.float) for m in transposed_mu_k]
                        # 우선은 오류 가능성을 두고 원래 로직대로 진행 (예제 데이터는 길이 통일)
                        transposed_mu_k = list(map(list, zip(*[w_list if w_list else [0.0]*len(batch_mu_weights_k_list[0] if batch_mu_weights_k_list[0] else 0) for w_list in batch_mu_weights_k_list])))
                        batch['mu_weights_k'] = [torch.tensor(m, dtype=torch.float) for m in transposed_mu_k]
                    else: # All non-empty lists have the same length
                        # Handle cases where some lists might be empty but others are not (pad empty ones)
                        expected_len = non_empty_lengths[0]
                        processed_mu_list = []
                        for w_list in batch_mu_weights_k_list:
                            if not w_list and expected_len > 0 : # list is empty, but should have items
                                processed_mu_list.append([0.0] * expected_len) # pad with zeros
                            else:
                                processed_mu_list.append(w_list)
                        
                        if not processed_mu_list or not processed_mu_list[0]: # All lists were empty initially or became empty after processing
                             batch['mu_weights_k'] = []
                        else:
                            transposed_mu_k = list(map(list, zip(*processed_mu_list)))
                            batch['mu_weights_k'] = [torch.tensor(m, dtype=torch.float) for m in transposed_mu_k]

                except Exception as e:
                    print(f"Error processing mu_weights_k: {e}. Setting to empty list.")
                    batch['mu_weights_k'] = []
            else:
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