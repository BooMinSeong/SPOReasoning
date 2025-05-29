#!/usr/bin/env python
import os
import logging
import torch
from datasets import load_from_disk, load_dataset
from vllm import LLM
from score import score
from argparse import ArgumentParser

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def args_parser():
    parser = ArgumentParser(description="LLM Inference")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct",)
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--num_return_sequences", type=int, default=10, help="Number of sequences to return")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    return parser.parse_args()

def main():
    args = args_parser()
    
    logger.info(f"Inf Start")
    dataset_save_name = f"MATH_testset_t{args.temperature}_n{args.num_return_sequences}"
    
    # load for full test
    dataset = load_dataset("HuggingFaceH4/MATH-500",split="test")
    
    logger.info("load dataset!!")


    # 예시 instruction (원하는 문구로 수정 가능)
    instruction: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: "

    # GPU 개수를 확인하고, LLM 인스턴스 생성
    num_gpus = torch.cuda.device_count()
    model_name = args.model_path
    save_path = os.path.join("results",'datas',model_name.split("/")[-2],model_name.split("/")[-1], dataset_save_name)

    instrcut_model = True if "Instruct" in model_name else False
    logger.info(f"instruct model: {instrcut_model}")
    
    llm = LLM(
        model = model_name,
        gpu_memory_utilization=0.8,  # 필요에 따라 조정
        enable_prefix_caching=True,
        seed=42,
        tensor_parallel_size=num_gpus if num_gpus > 0 else 1,
    )
    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = args.temperature
    sampling_params.max_tokens = args.max_tokens
    sampling_params.n=args.num_return_sequences

    logger.info("load model")

    def generate_response(batch):
        # batch는 딕셔너리로, 각 key에 대해 리스트 형태로 값이 들어있습니다.
        conversations = [
            [
                {
                    "role": "user",
                    "content": instruction +q                
                }
            ]
            for q in batch["problem"]
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
            responses = [error_response] * len(batch['problem']) # 배치 크기만큼 오류 응답 생성

        return {"completions": responses}


    # 데이터셋 전체에 대해 배치 처리로 응답 생성 수행
    dataset = dataset.map(
        generate_response, 
        batched=True, 
        batch_size=320,
        desc=f"generate answers"
        )
    # 멈추는 에러 핸들링 위해 중간 저장
    dataset.save_to_disk(save_path)

    dataset = score(dataset)
    

    dataset.save_to_disk(save_path)
    logger.info("Done 🔥!")
    
if __name__ == "__main__":
    main()
