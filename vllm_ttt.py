#!/usr/bin/env python
import os
import logging
import torch
from datasets import load_from_disk, concatenate_datasets, Dataset
from vllm import LLM
from score import score, json_parse
from utils.utils import set_seed
from argparse import ArgumentParser

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANG_MAP = {
    "ko": "Korean",
    "en": "English",
}


def parse_args():
    parser = ArgumentParser(description="VLLM Inference Script")
    parser.add_argument("--batch_size", type=int, default=256,)
    parser.add_argument("--num_shards", type=int, default=10, help="Number of shards to split the dataset into.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it", help="Name of the model to use.")
    parser.add_argument("--dataset_name", type=str, default="ko-en_translation_source", help="Name of the dataset to use.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the results.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate.")
    parser.add_argument("--num_return_sequences", type=int, default=16, help="Number of sequences to return.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization for the model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed)

    # load for full test
    dataset_save_name = f"ko_eng_translation_reasoning"
    dataset = load_from_disk(f"datas/{args.dataset_name}")
    dataset.shuffle(seed=args.seed)
    # dataset = dataset.select(range(20000)) # for test
    num_shards = args.num_shards
    logger.info("load dataset!!")


    # Example instruction (can be modified as needed)
    instruction: str = """### You are a good {src_lang}-{trg_lang} translator.
### Translate this from {src_lang} to {trg_lang} translation:

// Before providing the translation, analyze the source text thoroughly and prepare a detailed explanation of the translation process.
// The 'Reasoning' field in the output should describe the translation process step-by-step, focusing on understanding the source, breaking it down, making translation choices, and handling linguistic differences.

{src_lang}: {src}
{trg_lang}:

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":

```json
{{
    "Reasoning": string,  // The reasoning behind the translation. Write down the reasoning in detail, covering:
                          // 1. Initial understanding of the source text's meaning and context.
                          // 2. Breaking down the source text into key components or phrases.
                          // 3. Explanation of translation choices for specific words/phrases, considering nuance, grammar, and target language naturalness.
                          // 4. How grammatical structures were adapted from source to target language.
                          // 5. Any difficulties encountered (e.g., idioms, ambiguity) and how they were resolved.
                          // 6. How the final translation was assembled and refined.
    "Translation": string  // The final translation. Please use escape characters for the quotation marks in the sentence.
}}"""

    # GPU 개수를 확인하고, LLM 인스턴스 생성
    num_gpus = torch.cuda.device_count()
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = args.model_name
    instrcut_model = True if "Instruct" in model_name else False
    logger.info(f"instruct model: {instrcut_model}")
    
    llm = LLM(
        model = model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,  # 필요에 따라 조정
        seed=args.seed,
        tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
    )
    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = args.temperature
    sampling_params.max_tokens = args.max_tokens
    sampling_params.n=args.num_return_sequences

    logger.info("load model")
    # 데이터셋의 각 배치에 대해 응답 생성 함수
    def generate_response(batch):
        # batch는 딕셔너리로, 각 key에 대해 리스트 형태로 값이 들어있습니다.
        conversations = [
            [
                {
                    "role": "user",
                    "content": instruction.format(
                        src_lang=LANG_MAP[src_lang],
                        trg_lang=LANG_MAP[trg_lang],
                        src=src,
                    )    
                }
            ]
            for src_lang, trg_lang, src in zip(batch['src_lang'], batch['trg_lang'], batch['src'])
        ]
        try:
            outputs = llm.chat(messages=conversations,
                            sampling_params=sampling_params,
                            use_tqdm=True)

            responses = [[output.text for output in batch.outputs] for batch in outputs]
        except Exception as e:
            logger.error(f"Error during llm.chat in a batch: {e}")
            # 오류 발생 시, 해당 배치의 모든 항목에 대해 n개의 오류 메시지/None 반환
            error_response = [f"Error: {e}"] * sampling_params.n
            responses = [error_response] * len(batch['src']) # 배치 크기만큼 오류 응답 생성

        return {"translation": responses}


    # 데이터셋 전체에 대해 배치 처리로 응답 생성 수행
    # 안정성을 위해 sharding을 통해 데이터셋을 나누고, 각 샤드에 대해 처리
    model_folder_name = model_name.split("/")[-1]
    temp_shard_dir = os.path.join("results", model_folder_name, dataset_save_name + "_shards")
    os.makedirs(temp_shard_dir, exist_ok=True)
    
    processed_shard_paths = []
    # --- 샤드 처리 루프 ---
    for i in range(num_shards):
        shard_save_path = os.path.join(temp_shard_dir, f"shard_{i}")
        logger.info(f"--- Processing Shard {i + 1} / {num_shards} ---")

        # 현재 샤드 가져오기 (원본 데이터셋에서)
        current_shard = dataset.shard(num_shards=num_shards, index=i)
        logger.info(f"Shard {i} size: {len(current_shard)}")

        # 이미 처리된 샤드가 있다면 건너뛰기 (선택적 재시작 기능)
        if os.path.exists(shard_save_path):
            logger.warning(f"Shard {i} already processed. Skipping generation. Loading from: {shard_save_path}")
            processed_shard_paths.append(shard_save_path)
            continue # 다음 샤드로 이동

        # 데이터셋의 각 배치에 대해 응답 생성
        try:
            processed_shard = current_shard.map(
                generate_response,
                batched=True,
                batch_size=args.batch_size, # 배치 크기 확인
                desc=f"Generating answers for Shard {i}"
            )
            # 처리된 샤드 저장
            processed_shard.save_to_disk(shard_save_path)
            processed_shard_paths.append(shard_save_path)
            logger.info(f"Shard {i} processed and saved to {shard_save_path}")

        except Exception as e:
            logger.error(f"Error processing Shard {i}: {e}. This shard will be skipped.")
            # 실패한 샤드는 processed_shard_paths에 추가하지 않음

    # --- 데이터셋 통합 ---
    logger.info("--- Concatenating Processed Shards ---")
    if not processed_shard_paths:
        logger.error("No shards were processed successfully. Exiting.")
        return

    all_processed_datasets = []
    for path in processed_shard_paths:
        try:
            loaded_shard = load_from_disk(path)
            all_processed_datasets.append(loaded_shard)
            logger.info(f"Loaded processed shard from: {path}")
        except Exception as e:
            logger.error(f"Failed to load processed shard from {path}: {e}")

    if not all_processed_datasets:
        logger.error("Failed to load any processed shards. Cannot proceed.")
        return

    # 데이터셋 통합
    final_dataset = concatenate_datasets(all_processed_datasets)
    logger.info(f"Successfully concatenated {len(all_processed_datasets)} shards. Final dataset size: {len(final_dataset)}")

    # --- 스코어링 및 최종 저장 ---
    logger.info("--- Scoring Final Dataset ---")
    try:
        # final_dataset_scored = score(final_dataset)
        final_dataset_scored = json_parse(final_dataset)
        
        final_save_path = os.path.join("results", model_folder_name, dataset_save_name)
        final_dataset_scored.save_to_disk(final_save_path)
        logger.info(f"Final scored dataset saved to: {final_save_path}")

        # 임시 샤드 파일/디렉토리 삭제 (선택 사항)
        # import shutil
        # logger.info(f"Deleting temporary shard directory: {temp_shard_dir}")
        # shutil.rmtree(temp_shard_dir)

    except Exception as e:
         logger.error(f"Error during final scoring or saving: {e}")
         logger.info(f"Concatenated (but unscored) dataset might be available if intermediate steps succeeded.")


    logger.info("Done 🔥!")
    
if __name__ == "__main__":
    main()