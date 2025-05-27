#!/usr/bin/env python
import os
import logging
import torch
from datasets import load_from_disk
from vllm import LLM
from score import score
from utils.utils import PromptFormatter, set_seed
from argparse import ArgumentParser

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def args_parser():
    parser = ArgumentParser(description="LLM Inference")
    parser.add_argument("--start_bin", type=int, default=1, help="Start bin for filtering dataset")
    return parser.parse_args()

def main():
    args = args_parser()
    set_seed(42)
    
    start_bin = args.start_bin
    logger.info(f"Start bin: {start_bin}")
    dataset_save_name = f"MATH_w_cot"
    
    # load for full test
    dataset_save_name = f"MATH_w_cot_index_response_eval_total"
    dataset = load_from_disk("MATH_w_llama3_70B")
    # dataset = dataset.select(range(10000)) # for test
    
    logger.info("load dataset!!")


    # 예시 instruction (원하는 문구로 수정 가능)
    instruction: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: "

    # GPU 개수를 확인하고, LLM 인스턴스 생성
    num_gpus = torch.cuda.device_count()
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name = "meta-llama/Llama-3.1-8B"
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
    sampling_params.temperature = 1.0
    sampling_params.max_tokens = 4096

    logger.info("load model")
    # 데이터셋의 각 배치에 대해 응답 생성 함수
#    def generate_response(batch):
#        # batch는 딕셔너리로, 각 key에 대해 리스트 형태로 값이 들어있습니다.
#        if instrcut_model:
#            prompts = [
#                PromptFormatter.format_instruct(instruction, q,solution="" )
#                for q in batch["input_question"]
#            ]
#
#        outputs = llm.generate(prompts,sampling_params )
#
#        responses = [output.outputs[0].text for output in outputs]
#        return {"completions": responses}

    def generate_response(batch):
        # batch는 딕셔너리로, 각 key에 대해 리스트 형태로 값이 들어있습니다.
        conversations = [
            [
                {
                    "role": "user",
                    "content": instruction +q                
                }
            ]
            for q in batch["input_question"]
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

        return {"completions": responses}


    # 데이터셋 전체에 대해 배치 처리로 응답 생성 수행
    dataset = dataset.map(
        generate_response, 
        batched=True, 
        batch_size=256,
        desc=f"generate answers"
        )

    dataset = score(dataset)    
    save_path = os.path.join("results",model_name.split("/")[-1], dataset_save_name)

    dataset.save_to_disk(save_path)
    logger.info("Done 🔥!")
    
if __name__ == "__main__":
    main()
