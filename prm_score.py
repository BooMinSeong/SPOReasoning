#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk, Dataset as HFDataset
from accelerate import Accelerator
import numpy as np
import torch
from tqdm import tqdm
import argparse
import json
import time
import os
import sys
import re
import logging
from functools import partial

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="PRM Scoring for Multiple Completions")
    parser.add_argument("--reward_name_or_path", type=str, default='RLHFlow/Llama3.1-8B-PRM-Deepseek-Data', help="Reward model path")
    parser.add_argument("--input_dataset_path", type=str, required=True, help="Path to the input dataset directory (saved by vllm_inf.py)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for the scored dataset")
    parser.add_argument("--model_type", type=str, choices=["Mistral", "Deepseek", "Llama"], default='Llama', help="Type of PRM model, determines step delimiter ('Mistral' for 'ки\\n', 'Deepseek'/'OtherLlama' for '\\n\\n')")
    # num_n is mainly for output naming to be consistent with original script, actual N comes from data
    parser.add_argument("--num_n_label", type=int, default=10, help="Informative N for output file naming, corresponds to num_return_sequences used for generation.")
    parser.add_argument("--batch_size_per_device", type=int, default=2, help="Batch size for dataset.map processing per device")
    return parser.parse_args()

def add_prm_scores_to_batch(batch, prm_model, prm_tokenizer, prm_candidate_tokens, prm_args, prm_device):
    """
    Calculates PRM scores for each completion in a batch of data.
    Each item in the batch can have multiple completions.
    """
    batch_prompts = batch['input_question'] # Key from your vllm_inf.py dataset
    batch_completions_list = batch['completions'] # Key from your vllm_inf.py dataset

    all_samples_avg_scores = []
    all_samples_step_scores = []

    for i in range(len(batch_prompts)):
        prompt = batch_prompts[i]
        completions_for_sample = batch_completions_list[i] # This is a list of N completion strings

        sample_avg_scores_for_n_completions = []
        sample_step_scores_for_n_completions = []

        for completion_text in completions_for_sample:
            single_ans_step_values = []
            current_conversation_for_rm = [] # Resets for each completion

            # Determine step delimiter based on PRM model type
            if prm_args.model_type == "Mistral":
                ans_steps = completion_text.split("ки\n")
            elif prm_args.model_type in ["Deepseek", "Llama"]: # Common case for Llama-like models
                ans_steps = completion_text.split("\n\n")
            else: # Fallback, should ideally be configured or error
                logger.warning(f"Unknown model_type '{prm_args.model_type}', defaulting to '\\n\\n' delimiter.")
                ans_steps = completion_text.split("\n\n")
            
            ans_steps = [step.strip() for step in ans_steps if step.strip()]

            if not ans_steps: # Handle completions that result in no scorable steps
                sample_avg_scores_for_n_completions.append(0.0)
                sample_step_scores_for_n_completions.append([])
                continue

            for k, step_text in enumerate(ans_steps):
                if k == 0:
                    full_step_input_text = prompt + " " + step_text
                else:
                    full_step_input_text = step_text # Subsequent steps are just the step text

                current_conversation_for_rm.append({"content": full_step_input_text, "role": "user"})
                current_conversation_for_rm.append({"content": "+", "role": "assistant"}) # PRM scoring convention

                try:
                    input_ids = prm_tokenizer.apply_chat_template(
                        current_conversation_for_rm,
                        return_tensors="pt"
                    ).to(prm_device)

                    with torch.no_grad():
                        logits_output = prm_model(input_ids).logits
                        
                        # Ensure the sequence is long enough for the -3 index
                        if logits_output.shape[1] < 3:
                            logger.warning(f"Sequence length {logits_output.shape[1]} too short for -3 index. Assigning score 0. Prompt: {prompt[:50]}, Step: {step_text[:50]}")
                            positive_score = 0.0 
                        else:
                            # Get logits for '+' and '-' tokens at the designated position
                            relevant_logits = logits_output[:, -3, prm_candidate_tokens]
                            scores_softmax = relevant_logits.softmax(dim=-1)
                            positive_score = scores_softmax[:, 0].detach().to('cpu', dtype=torch.float32).item() # Prob of '+'
                    single_ans_step_values.append(positive_score)
                except Exception as e:
                    logger.error(f"Error during PRM scoring for a step: {e}. Prompt: {prompt[:50]}, Step: {step_text[:50]}")
                    single_ans_step_values.append(-1.0) # Error score

            sample_step_scores_for_n_completions.append(single_ans_step_values)
            if single_ans_step_values and not all(s == -1.0 for s in single_ans_step_values):
                valid_scores = [s for s in single_ans_step_values if s != -1.0]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            else:
                avg_score = 0.0 # Default if no valid steps or all errors
            sample_avg_scores_for_n_completions.append(avg_score)
        
        all_samples_avg_scores.append(sample_avg_scores_for_n_completions)
        all_samples_step_scores.append(sample_step_scores_for_n_completions)

    # These will be new columns in the dataset
    # Name columns uniquely based on the reward model
    reward_model_name_slug = prm_args.reward_name_or_path.split('/')[-1].replace('-', '_')
    return {
        f"{reward_model_name_slug}_avg_scores": all_samples_avg_scores,
        f"{reward_model_name_slug}_step_scores": all_samples_step_scores
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
            torch_dtype=torch.bfloat16 # Or float16, adjust based on your GPU and model
        )
        model = model.to(accelerator.device).eval()
    except Exception as e:
        logger.error(f"Failed to load PRM model or tokenizer: {e}")
        return

    tokenizer.padding_side = "right" # Or "left" depending on model training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


    # '+' and '-' token IDs for scoring
    # Ensure not to add special tokens like <s> or </s> when encoding just '+' or '-'
    plus_tag_id = tokenizer.encode('+', add_special_tokens=False)
    minus_tag_id = tokenizer.encode('-', add_special_tokens=False)

    if not plus_tag_id or not minus_tag_id:
        logger.error("Could not encode '+' or '-' tokens correctly. Check tokenizer.")
        return
    plus_tag_id = plus_tag_id[0] # Assuming simple tokens
    minus_tag_id = minus_tag_id[0]
    
    candidate_tokens = torch.tensor([plus_tag_id, minus_tag_id], device=accelerator.device)
    logger.info(f"Candidate tokens for PRM scoring (+, -): [{plus_tag_id}, {minus_tag_id}]")

    # Load the input dataset
    logger.info(f"Loading input dataset from: {args.input_dataset_path}")
    try:
        # Ensure dataset is loaded by all processes before sharding, or handle appropriately
        with accelerator.main_process_first():
            dataset_to_process = load_from_disk(args.input_dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.input_dataset_path}: {e}")
        return
    
    logger.info(f"Dataset loaded. Number of samples: {len(dataset_to_process)}")

    # Shard the dataset for distributed processing
    sharded_dataset = dataset_to_process.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
    logger.info(f"Process {accelerator.process_index}/{accelerator.num_processes} processing {len(sharded_dataset)} samples.")

    # Prepare the map function with necessary arguments
    map_fn = partial(add_prm_scores_to_batch,
                     prm_model=model,
                     prm_tokenizer=tokenizer,
                     prm_candidate_tokens=candidate_tokens,
                     prm_args=args,
                     prm_device=accelerator.device)

    logger.info(f"Starting PRM scoring using .map() on process {accelerator.process_index}...")
    # Apply the map function to the sharded dataset
    updated_sharded_dataset = sharded_dataset.map(
        map_fn,
        batched=True,
        batch_size=args.batch_size_per_device,
        desc=f"PRM Scoring (Process {accelerator.process_index})"
    )
    logger.info(f"PRM scoring complete on process {accelerator.process_index}.")

    # Gather all processed shards (list of dictionaries)
    list_of_dicts_shard = [updated_sharded_dataset[i] for i in range(len(updated_sharded_dataset))]
    
    logger.info(f"Gathering results from all processes on process {accelerator.process_index}...")
    all_processed_list_of_dicts = accelerator.gather_object(list_of_dicts_shard)
    logger.info(f"Result gathering complete on process {accelerator.process_index}.")


    if accelerator.is_main_process:
        logger.info("Main process is reconstructing the final dataset.")
        final_data_list = []
        for sublist in all_processed_list_of_dicts:
            final_data_list.extend(sublist)
        
        # Ensure correct order if sharding and gathering might reorder.
        # If original dataset had an 'id' or 'index' field, sorting by it is safest.
        # For now, assuming gather_object and shard preserve order relative to original.
        # A robust way is to sort `final_data_list` based on an original index if available.
        # If your dataset does not have a unique identifier per row that persists through sharding,
        # you might need to add one before sharding if order is critical and not guaranteed.

        final_dataset = HFDataset.from_list(final_data_list)
        logger.info(f"Final dataset reconstructed. Total samples: {len(final_dataset)}")

        # Define output path
        reward_model_name_slug = args.reward_name_or_path.split('/')[-1]
        output_dataset_name = f"{reward_model_name_slug}_scored_N{args.num_n_label}"
        output_dataset_dir = os.path.join(args.output_dir, output_dataset_name)
        
        os.makedirs(args.output_dir, exist_ok=True)
        # No need to create output_dataset_dir, save_to_disk will do it.
        
        logger.info(f"Saving scored dataset to: {output_dataset_dir}")
        final_dataset.save_to_disk(output_dataset_dir)
        logger.info(f"Successfully saved scored dataset to {output_dataset_dir} 🚀")

    accelerator.wait_for_everyone()
    logger.info("All processes finished.")

if __name__ == "__main__":
    main()