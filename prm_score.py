#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, Dataset as HFDataset
from accelerate import Accelerator
import torch
from tqdm import tqdm
import argparse
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="PRM Scoring for Multiple Completions")
    parser.add_argument("--reward_name_or_path", type=str, default='RLHFlow/Llama3.1-8B-PRM-Deepseek-Data', help="Reward model path")
    parser.add_argument("--input_dataset_path", type=str, default='results/Llama-3.2-1B-Instruct/MATH_w_cot_parsed', help="Path to the input dataset directory (saved by vllm_inf.py)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for the scored dataset")
    parser.add_argument("--model_type", type=str, choices=["Mistral", "Deepseek", "Llama"], default='Llama', help="Type of PRM model, determines step delimiter ('Mistral' for 'ки\\n', 'Deepseek'/'OtherLlama' for '\\n\\n')")
    parser.add_argument("--num_n_label", type=int, default=10, help="Informative N for output file naming, corresponds to num_return_sequences used for generation.")
    parser.add_argument("--batch_size_per_device", type=int, default=2, help="Batch size for processing per device")
    return parser.parse_args()

def add_prm_scores_to_batch(batch_data_dict, prm_model, prm_tokenizer, prm_candidate_tokens, prm_args, prm_device):
    batch_prompts = batch_data_dict['problem']
    batch_completions_list = batch_data_dict['completions']
    batch_all_samples_avg_scores = []
    batch_all_samples_step_scores = []

    for i in range(len(batch_prompts)):
        prompt = batch_prompts[i]
        completions_for_sample = batch_completions_list[i]
        sample_avg_scores_for_n_completions = []
        sample_step_scores_for_n_completions = []

        for completion_text in completions_for_sample:
            single_ans_step_values = []
            current_conversation_for_rm = []
            if prm_args.model_type == "Mistral":
                ans_steps = completion_text.split("ки\n")
            elif prm_args.model_type in ["Deepseek", "Llama"]:
                ans_steps = completion_text.split("\n\n")
            else:
                logger.warning(f"Unknown model_type '{prm_args.model_type}', defaulting to '\\n\\n' delimiter.")
                ans_steps = completion_text.split("\n\n")
            ans_steps = [step.strip() for step in ans_steps if step.strip()]

            if not ans_steps:
                sample_avg_scores_for_n_completions.append(0.0)
                sample_step_scores_for_n_completions.append([])
                continue

            for k, step_text in enumerate(ans_steps):
                if k == 0:
                    full_step_input_text = prompt + " " + step_text
                else:
                    full_step_input_text = step_text
                current_conversation_for_rm.extend([
                    {"content": full_step_input_text, "role": "user"},
                    {"content": "+", "role": "assistant"}
                ])
                try:
                    input_ids = prm_tokenizer.apply_chat_template(
                        current_conversation_for_rm, return_tensors="pt").to(prm_device)
                    with torch.no_grad():
                        logits_output = prm_model(input_ids).logits
                        if logits_output.shape[1] < 3:
                            logger.warning(f"Sequence length {logits_output.shape[1]} too short. Assigning score 0. P: {prompt[:30]}, S: {step_text[:30]}")
                            positive_score = 0.0 
                        else:
                            relevant_logits = logits_output[:, -3, prm_candidate_tokens]
                            scores_softmax = relevant_logits.softmax(dim=-1)
                            positive_score = scores_softmax[:, 0].detach().to('cpu', dtype=torch.float32).item()
                    single_ans_step_values.append(positive_score)
                except Exception as e:
                    logger.error(f"Error in PRM step scoring: {e}. P: {prompt[:30]}, S: {step_text[:30]}")
                    single_ans_step_values.append(-1.0)
                current_conversation_for_rm.pop() # Remove assistant "+"
                current_conversation_for_rm.pop() # Remove user current step text

            sample_step_scores_for_n_completions.append(single_ans_step_values)
            valid_scores = [s for s in single_ans_step_values if s != -1.0]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            sample_avg_scores_for_n_completions.append(avg_score)
        
        batch_all_samples_avg_scores.append(sample_avg_scores_for_n_completions)
        batch_all_samples_step_scores.append(sample_step_scores_for_n_completions)

    # reward_model_name_slug = prm_args.reward_name_or_path.split('/')[-1].replace('-', '_')
    return {
        f"prm_avg_scores": batch_all_samples_avg_scores,
        f"prm_step_scores": batch_all_samples_step_scores
    }

def main():
    args = parse_args()
    accelerator = Accelerator()

    logger.info(f"Using device: {accelerator.device}")
    logger.info(f"Loading PRM tokenizer: {args.reward_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
        logger.info(f"Loading PRM model: {args.reward_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.reward_name_or_path,
            torch_dtype=torch.bfloat16
        )
        model = model.to(accelerator.device).eval()
    except Exception as e:
        logger.error(f"Failed to load PRM model or tokenizer: {e}")
        return

    tokenizer.padding_side = "right" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    plus_tag_id = tokenizer.encode('+', add_special_tokens=False)[0]
    minus_tag_id = tokenizer.encode('-', add_special_tokens=False)[0]
    if not plus_tag_id or not minus_tag_id: # Check if encode returned empty or 0
        logger.error("Could not encode '+' or '-' tokens correctly.")
        return
    candidate_tokens = torch.tensor([plus_tag_id, minus_tag_id], device=accelerator.device)
    logger.info(f"Candidate tokens for PRM scoring (+, -): [{plus_tag_id}, {minus_tag_id}]")

    logger.info(f"Loading input dataset from: {args.input_dataset_path}")
    try:
        with accelerator.main_process_first():
             dataset_to_process = load_from_disk(args.input_dataset_path)
             dataset_to_process = dataset_to_process.shuffle(seed=42)  # Shuffle dataset for better distribution across processes
            #  dataset_to_process = dataset_to_process.select(range(5000))
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.input_dataset_path}: {e}")
        return
    logger.info(f"Dataset loaded. Number of samples: {len(dataset_to_process)}")

    num_total_samples = len(dataset_to_process)
    indices_for_this_process = list(range(num_total_samples))[accelerator.process_index::accelerator.num_processes]
    logger.info(f"Process {accelerator.process_index}/{accelerator.num_processes} processing {len(indices_for_this_process)} samples.")

    processed_items_this_process = []
    for i in tqdm(range(0, len(indices_for_this_process), args.batch_size_per_device),
                  desc=f"PRM Scoring (Process {accelerator.process_index})",
                  disable=not accelerator.is_local_main_process):
        batch_indices = indices_for_this_process[i:i+args.batch_size_per_device]
        if not batch_indices:
            continue
        current_batch_data = dataset_to_process[batch_indices]
        score_results = add_prm_scores_to_batch(
            current_batch_data, model, tokenizer, candidate_tokens, args, accelerator.device)
        
        num_items_in_batch = len(current_batch_data[next(iter(current_batch_data))])
        for j in range(num_items_in_batch):
            item_data = {key: current_batch_data[key][j] for key in current_batch_data}
            for score_col, scores_list in score_results.items():
                item_data[score_col] = scores_list[j]
            processed_items_this_process.append(item_data)

    logger.info(f"PRM scoring loop complete on process {accelerator.process_index}.")
    logger.info(f"Gathering results from all processes on process {accelerator.process_index}...")
    all_gathered_items = accelerator.gather(processed_items_this_process)
    logger.info(f"Result gathering complete on process {accelerator.process_index}.")

    if accelerator.is_main_process:
        logger.info("Main process is reconstructing the final dataset.")
        final_data_list = []

        if accelerator.num_processes == 1:
            # 단일 프로세스: all_gathered_items는 List[Dict] 형태 (이전 로그 기반)
            if all_gathered_items and isinstance(all_gathered_items, list) and \
               (not all_gathered_items or isinstance(all_gathered_items[0], dict)):
                final_data_list = all_gathered_items
            # 단일 프로세스지만 gather가 List[List[Dict]]를 반환한 경우 (이론상 표준 동작)
            elif all_gathered_items and isinstance(all_gathered_items, list) and \
                 len(all_gathered_items) == 1 and isinstance(all_gathered_items[0], list):
                final_data_list = all_gathered_items[0]
            else:
                logger.warning(f"Single process: all_gathered_items has unexpected structure ({type(all_gathered_items)}) or is empty. final_data_list might be incorrect.")
        else: # 다중 프로세스 (num_processes > 1)
            # all_gathered_items는 List[List[Dict]] 형태를 기대
            for sublist_from_process in all_gathered_items:
                if isinstance(sublist_from_process, list):
                    final_data_list.extend(sublist_from_process)
                else:
                    logger.error(f"Multiple processes: Expected a sublist, but got {type(sublist_from_process)}. Skipping this item.")
        
        logger.info(f"Total items in final_data_list after reconstruction: {len(final_data_list)}")
        
        if not final_data_list:
            logger.warning("The final data list is empty. No dataset will be saved.")
        elif not isinstance(final_data_list[0], dict):
            logger.error(f"CRITICAL Error: First item in final_data_list is not a dict (type: {type(final_data_list[0])}). Dataset creation aborted.")
            logger.error("Dumping first 5 items of final_data_list for inspection:")
            for k, item in enumerate(final_data_list[:5]):
                logger.error(f"Item {k}: Type={type(item)}, Value (partial)='{str(item)[:200]}'")
        else:
            try:
                final_dataset = HFDataset.from_list(final_data_list)
                logger.info(f"Final dataset reconstructed. Total samples: {len(final_dataset)}")

                slug = os.path.basename(args.input_dataset_path) 
                out_name = f"{slug}_scored"
                input_dir = os.path.dirname(args.input_dataset_path)
                out_dir = os.path.join(input_dir, out_name)
                os.makedirs(args.output_dir, exist_ok=True)
                
                logger.info(f"Saving scored dataset to: {out_dir}")
                final_dataset.save_to_disk(out_dir)
                logger.info(f"Successfully saved scored dataset to {out_dir} 🚀")
            except Exception as e:
                logger.error(f"Error during final dataset creation or saving: {e}")
                logger.error("Dumping first 5 items of final_data_list (if available):")
                for k, item in enumerate(final_data_list[:5]):
                    logger.error(f"Item {k}: Type={type(item)}, Keys (if dict)='{list(item.keys()) if isinstance(item, dict) else 'N/A'}', Value (partial)='{str(item)[:200]}'")

    accelerator.wait_for_everyone()
    logger.info("All processes finished.")

if __name__ == "__main__":
    main()