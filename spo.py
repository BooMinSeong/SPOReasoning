import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM

class SPOLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.01,
                 beta: float = 0.1,
                 gamma: float = 0.01,
                 eta_decay: float = 1.0,
                 mu_scale_factor: float = 2.0,
                 reference_model: Optional[nn.Module] = None,
                 use_global_kl: bool = False,
                 use_spo_mu: bool = False,
                ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta_decay = eta_decay
        self.mu_scale_factor = mu_scale_factor
        self.use_global_kl = use_global_kl
        self.mu_inner_factor = 0.8
        self.use_spo_mu = use_spo_mu

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

    # def _calculate_mu_k(self,
    #                     current_k: int,
    #                     scores_for_current_sample_ranked: torch.Tensor, 
    #                    ) -> torch.Tensor:
    #     scores_in_C_k = scores_for_current_sample_ranked[current_k:]
    #     sum_scores_in_C_k = torch.sum(scores_in_C_k)
    #     V_ik_current = sum_scores_in_C_k**self.gamma
        
    #     # V_ik_current를 현재 샘플의 평균 V값과 비교
    #     impact_score = V_ik_current - torch.mean(sum_scores_in_C_k) # 변경된 인자 이름 사용
    #     overall_quality_weight = self.mu_scale_factor * torch.sigmoid(impact_score)
        
    #     final_mu_k = (self.eta_decay**current_k) * overall_quality_weight
    #     return final_mu_k
    
    # simple version
    def _calculate_mu_k(self,
                        current_k: int,
                        scores_for_current_sample_ranked: torch.Tensor, 
                       ) -> torch.Tensor:
        scores_in_C_k = scores_for_current_sample_ranked[current_k:]
        sum_scores_in_C_k = torch.sum(scores_in_C_k)
        # V_ik_current를 현재 샘플의 평균 V값과 비교
        impact_score = self.mu_inner_factor*(sum_scores_in_C_k**self.gamma)
        overall_quality_weight = self.mu_scale_factor * torch.sigmoid(impact_score)
        
        final_mu_k = (self.eta_decay**current_k) * overall_quality_weight
        return final_mu_k
    
    def _calculate_mu_k_spo(self,
                        current_k: int,
                        scores_for_current_sample_ranked: torch.Tensor, 
                       ) -> torch.Tensor:
        scores_in_C_k = scores_for_current_sample_ranked[current_k:]
        scores_in_C_k = torch.exp(scores_in_C_k)  # Exponentiate to get actual scores
        sum_scores_in_C_k = torch.sum(scores_in_C_k)
        # V_ik_current를 현재 샘플의 평균 V값과 비교
        impact_score = self.mu_inner_factor*(sum_scores_in_C_k**self.gamma)
        overall_quality_weight = self.mu_scale_factor * torch.sigmoid(impact_score)
        
        final_mu_k = (self.eta_decay**current_k) * overall_quality_weight
        return final_mu_k

    def _calculate_tokenwise_kl_on_sequences(self,
                                             policy_logits: torch.Tensor,
                                             input_ids: torch.Tensor,
                                             attention_mask: torch.Tensor,
                                             labels: torch.Tensor
                                            ) -> torch.Tensor:
        # self.reference_model이 None인 경우는 forward에서 beta > 0일 때 미리 체크함.
        current_device = policy_logits.device
        if self.reference_model.device != current_device:
            self.reference_model.to(current_device)
    
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
               ):
        
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
                    # if self.use_spo_mu:
                    #     mu_k = self._calculate_mu_k_spo(
                    #         k, current_sample_log_probs_policy_ranked
                    #         # k, current_sample_scores_ranked
                    #     )
                    # else:
                    #     mu_k = self._calculate_mu_k(
                    #         # k, current_sample_log_probs_policy_ranked
                    #         k, current_sample_scores_ranked
                    #     )
                    # print(f"::: mu_k for sample {i}, k={k}: {mu_k.item()}")
                    term_k = -(1.0 / self.alpha) * term_k_log_ratio * 1
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
        # 디버깅을 위해 각 손실 요소 출력
        print(f"Final Preference Loss: {final_preference_loss.item()}")
        print(f"KL Divergence Loss (raw): {kl_divergence_loss.item()}")

        return total_loss
    
# --- 2. Custom Trainer Class (compute_loss 수정) ---
class CustomSPOTrainer(Trainer):
    def __init__(self, spo_loss_fn: SPOLoss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spo_loss_fn = spo_loss_fn


    # compute_loss 메서드 시그니처 수정
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # <-- num_items_in_batch 추가
        model.train()


        loss = self.spo_loss_fn.forward(model, inputs)
        loss = loss/self.args.gradient_accumulation_steps

        return (loss, None) if return_outputs else loss


# --- SPODataCollator 수정 ---
import torch
from typing import List, Dict, Any, Optional

# Assuming you have your tokenizer loaded, e.g., from transformers
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf") # Example

class SPODataCollator:
    def __init__(self, tokenizer: Any, instruction: str, max_length: int = 512, model_type: str = "decoder"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction.strip()
        self.model_type = model_type # "decoder" or "encoder-decoder"

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Warning: tokenizer.pad_token was None, set to eos_token: {self.tokenizer.pad_token}")
            else:
                try:
                    # Try adding a common pad token. If your tokenizer has a different one, adjust.
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    print(f"Warning: Tokenizer did not have a pad_token. Added new one ('[PAD]') and set it as pad_token.")
                except Exception as e:
                    raise ValueError(
                        "Tokenizer needs a pad_token or eos_token, or be configurable to add one. "
                        f"Attempting to add '[PAD]' failed: {e}"
                    )
        
        # Some tokenizers might have pad_token set but not pad_token_id
        if self.tokenizer.pad_token_id is None and self.tokenizer.pad_token is not None:
             self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        # For decoder-only models, padding on the left is often preferred during training.
        # However, since we are manually padding to the right in this collator,
        # this setting primarily affects tokenizer's own padding if used directly elsewhere.
        # For apply_chat_template, direct padding control is less common.
        if self.model_type == "decoder" and self.tokenizer.padding_side != "right":
            print(f"Info: For decoder models, right padding is applied by this collator. Current tokenizer.padding_side='{self.tokenizer.padding_side}'.")
            # tokenizer.padding_side = "left" # if you prefer left padding and tokenizer supports it for other ops


    def _apply_chat_template_for_sequence(self, prompt_text: str, response_text: Optional[str] = None) -> List[int]:
        """
        Applies chat template for a prompt and an optional response.
        If response_text is None, it's for calculating prompt length (add_generation_prompt=True).
        If response_text is provided, it's for the full sequence (add_generation_prompt=False).
        """
        messages = []
        if self.instruction: # Add instruction as a system prompt or part of user prompt
            # Option 1: Add as a separate system message if tokenizer supports it well
            # messages.append({"role": "system", "content": self.instruction})
            # messages.append({"role": "user", "content": prompt_text})
            # Option 2: Prepend to user prompt (simpler, more general)
            full_prompt = self.instruction + " " + prompt_text if self.instruction else prompt_text
            messages.append({"role": "user", "content": full_prompt})
        else:
            messages.append({"role": "user", "content": prompt_text})

        if response_text is not None:
            messages.append({"role": "assistant", "content": response_text})
            # For training, we provide the assistant's message, so no generation prompt needed.
            add_gen_prompt = False
        else:
            # To get the length of the prompt *including* template tokens leading to assistant's turn
            add_gen_prompt = True

        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                max_length=self.max_length,
                truncation=True,
                add_generation_prompt=add_gen_prompt, # Key difference
                # return_tensors="pt", # We'll convert to tensor later after padding
                return_attention_mask=False # We create it manually after padding
            )
            # If apply_chat_template returns a list of lists (e.g. for some tokenizers when not returning tensors)
            if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            return token_ids
        except Exception as e:
            print(f"Error applying chat template. Messages: {messages}, add_generation_prompt: {add_gen_prompt}")
            print(f"Tokenizer: {self.tokenizer}")
            # Try to get more info if it's a common Hugging Face tokenizer
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is None:
                print("Critical: tokenizer.chat_template is None. You need to set a chat template for this tokenizer first. "
                      "Example: tokenizer.chat_template = \"{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}\"")
            raise e


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {}

        first_item_type = 'ranked'
        if first_item_type is None: raise ValueError("Feature must have a 'type' key.")

        num_responses_per_sample = 0
        if first_item_type == "ranked":
            if not features[0].get("completions") or not isinstance(features[0]["completions"], list):
                raise ValueError("First feature of type 'ranked' must have a list of 'completions'.")
            num_responses_per_sample = len(features[0]["completions"])
            if num_responses_per_sample == 0:
                raise ValueError("Type 'ranked' features must have at least one completion.")
        elif first_item_type == "best_of_n":
            num_responses_per_sample = len(features[0].get("completions", []))


        global_max_seq_len_in_batch = 0
        for feature in features:
            prompt_text = feature["problem"]
            responses_text = feature["completions"]

            if first_item_type == "ranked" and len(responses_text) != num_responses_per_sample:
                raise ValueError(f"Data inconsistency: Feature '{feature.get('problem','N/A')}' has {len(responses_text)} completions, expected {num_responses_per_sample}.")

            for response_text in responses_text:
                full_token_ids = self._apply_chat_template_for_sequence(prompt_text, response_text)
                global_max_seq_len_in_batch = max(global_max_seq_len_in_batch, len(full_token_ids))
        
        if global_max_seq_len_in_batch == 0 and num_responses_per_sample > 0 : # if num_responses_per_sample is 0, this is fine
             raise ValueError("global_max_seq_len_in_batch is 0. This might happen if all completions are empty or chat template produces empty output.")
        if global_max_seq_len_in_batch == 0: global_max_seq_len_in_batch = 1 # Avoid issues with empty tensors if K=0

        batch_candidate_input_ids = []
        batch_candidate_attention_mask = []
        batch_candidate_labels = []

        if first_item_type == "best_of_n":
            batch_chosen_index_in_candidates = []
            batch_mu_weights_scalar = []
        elif first_item_type == "ranked":
            batch_mu_weights_k_list = []

        for feature in features:
            prompt_text = feature["problem"]
            responses_text = feature["completions"]

            # Determine the length of the tokenized prompt (including instruction and template)
            # to know where to start labels.
            # Here, response_text is None, so _apply_chat_template_for_sequence uses add_generation_prompt=True
            prompt_only_token_ids = self._apply_chat_template_for_sequence(prompt_text, None)
            len_prompt_tokens_in_full = len(prompt_only_token_ids)
            
            # Ensure prompt tokens are not empty, otherwise masking labels might be problematic.
            if len_prompt_tokens_in_full == 0 and any(responses_text):
                 print(f"Warning: Prompt tokenization resulted in zero tokens for prompt: '{prompt_text}'. This might lead to incorrect label masking.")


            sample_cand_ids, sample_cand_attn, sample_cand_labels = [], [], []

            for response_text in responses_text:
                # Tokenize full sequence (prompt + response) using chat template
                # Here, response_text is provided, so _apply_chat_template_for_sequence uses add_generation_prompt=False
                full_token_ids = self._apply_chat_template_for_sequence(prompt_text, response_text)
                
                current_seq_len = len(full_token_ids)
                padding_len = global_max_seq_len_in_batch - current_seq_len
                
                final_ids_list = full_token_ids + [self.tokenizer.pad_token_id] * padding_len
                final_attn_list = [1] * current_seq_len + [0] * padding_len

                final_ids = torch.tensor(final_ids_list, dtype=torch.long)
                final_attn = torch.tensor(final_attn_list, dtype=torch.long)
                
                final_labels = final_ids.clone()
                
                # Mask prompt tokens (all tokens up to the end of the templated prompt)
                # This includes user message, system message (if any), and template tokens for assistant's turn.
                # Ensure mask_end_idx does not exceed the actual sequence length before padding.
                mask_end_idx = min(len_prompt_tokens_in_full, current_seq_len)
                final_labels[:mask_end_idx] = -100
                
                # Mask padding tokens (tokens from end of actual sequence to global_max_seq_len_in_batch)
                if padding_len > 0:
                    final_labels[current_seq_len:] = -100
                
                sample_cand_ids.append(final_ids)
                sample_cand_attn.append(final_attn)
                sample_cand_labels.append(final_labels)
            
            if not sample_cand_ids and num_responses_per_sample > 0 :
                raise ValueError(f"Feature '{prompt_text}' yielded no tokenized responses despite expecting K={num_responses_per_sample}.")
            
            # Stack K responses for this sample: (K, global_max_seq_len_in_batch)
            # Handle cases where num_responses_per_sample might be 0 (e.g., for best_of_n with no completions)
            if sample_cand_ids:
                batch_candidate_input_ids.append(torch.stack(sample_cand_ids))
                batch_candidate_attention_mask.append(torch.stack(sample_cand_attn))
                batch_candidate_labels.append(torch.stack(sample_cand_labels))
            elif num_responses_per_sample > 0: # Expected completions but got none tokenized
                 raise ValueError(f"Logic error or empty tokenization: sample_cand_ids is empty for K={num_responses_per_sample} for prompt: {prompt_text}")
            else: # K=0, so append empty tensors of the correct shape
                batch_candidate_input_ids.append(torch.empty(0, global_max_seq_len_in_batch, dtype=torch.long))
                batch_candidate_attention_mask.append(torch.empty(0, global_max_seq_len_in_batch, dtype=torch.long))
                batch_candidate_labels.append(torch.empty(0, global_max_seq_len_in_batch, dtype=torch.long))


            if first_item_type == "best_of_n":
                batch_chosen_index_in_candidates.append(feature["chosen_idx"])
                batch_mu_weights_scalar.append(feature.get("mu_weight", 1.0)) # or prm_avg_scores
            elif first_item_type == "ranked":
                batch_mu_weights_k_list.append(feature["prm_avg_scores"])

        batch = {"type": [first_item_type] * len(features)}
        
        if not batch_candidate_input_ids:
             if num_responses_per_sample > 0 and features : # Only raise if we expected data
                raise ValueError("Batch candidate input_ids is empty after processing all features. Check data and tokenization.")
             # If features list was empty or K=0, it's possible to have empty lists here
             # Create empty tensors with appropriate dimensions for consistency if needed by downstream.
             # For K > 0 and features present, this indicates a problem.
             # If K=0, the output tensors will be (B, 0, seq_len)
             # The stacking below might fail if batch_candidate_input_ids is truly empty and B > 0.
             # Let's refine the condition for raising error:
             if features and num_responses_per_sample > 0 :
                 raise ValueError("No valid candidate input IDs found. Ensure features have valid completions.")
             elif not features: # No features, return minimal batch
                 return batch # Or handle as per training loop's expectation for empty batch

        # Ensure all tensors in the list have the same K dimension before stacking for B
        # This should be guaranteed by the num_responses_per_sample logic earlier
        # Stacking will create (B, K, global_max_seq_len_in_batch)
        # If K=0, then it becomes (B, 0, global_max_seq_len_in_batch)
        try:
            batch['candidate_input_ids'] = torch.stack(batch_candidate_input_ids)
            batch['candidate_attention_mask'] = torch.stack(batch_candidate_attention_mask)
            batch['candidate_labels'] = torch.stack(batch_candidate_labels)
        except RuntimeError as e:
            print("RuntimeError during stacking. This often means inconsistent tensor shapes.")
            for i, t in enumerate(batch_candidate_input_ids): print(f"Shape of input_ids {i}: {t.shape}")
            for i, t in enumerate(batch_candidate_attention_mask): print(f"Shape of attention_mask {i}: {t.shape}")
            for i, t in enumerate(batch_candidate_labels): print(f"Shape of labels {i}: {t.shape}")
            print(f"Expected K (num_responses_per_sample): {num_responses_per_sample}")
            raise e


        if first_item_type == "best_of_n":
            batch['chosen_index_in_candidates'] = torch.tensor(batch_chosen_index_in_candidates, dtype=torch.long)
            batch['prm_avg_scores'] = torch.tensor(batch_mu_weights_scalar, dtype=torch.float)
        elif first_item_type == "ranked":
            if batch_mu_weights_k_list:
                # Ensure all lists have K elements, pad if necessary (though ideally data is consistent)
                expected_k = num_responses_per_sample
                processed_mu_weights = []
                for mu_list in batch_mu_weights_k_list:
                    if len(mu_list) != expected_k:
                        # This case should ideally be an error or handled by padding if permissible
                        raise ValueError(f"Inconsistent number of mu_weights. Expected {expected_k}, got {len(mu_list)}. Data: {mu_list}")
                    processed_mu_weights.append(torch.tensor(mu_list, dtype=torch.float))
                
                if processed_mu_weights: # if K > 0
                    batch['prm_avg_scores'] = torch.stack(processed_mu_weights) # (B, K)
                elif expected_k > 0 : # K>0 but no weights processed, error
                    raise ValueError("No mu_weights processed for 'ranked' type with K > 0.")
                else: # K=0
                    batch['prm_avg_scores'] = torch.empty(len(features), 0, dtype=torch.float)

            elif num_responses_per_sample > 0 : # K > 0 but list is empty
                 raise ValueError("No valid mu_weights_k found for 'ranked' type with K > 0. Ensure features have valid prm_avg_scores lists.")
            else: # K=0
                batch['prm_avg_scores'] = torch.empty(len(features), 0, dtype=torch.float) # (B,0) tensor
                
        return batch