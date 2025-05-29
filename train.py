import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType

import logging
import json
import os
from datetime import datetime
import argparse

from spo import CustomSPOTrainer, SPOLoss, SPODataCollator




# --- 로깅 설정 ---
def setup_logging(base_log_dir="./logs",exp_name=""):
    """
    Sets up two loggers with separate subdirectories for system and metrics logs:
    - System logs in: base_log_dir/system/system_YYYYMMDD_HHMMSS.log
    - Metrics logs in: base_log_dir/metrics/metrics_YYYYMMDD_HHMMSS.jsonl
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 하위 디렉토리 경로 정의 및 생성 ---
    system_log_dir = os.path.join(base_log_dir, "system")
    metrics_log_dir = os.path.join(base_log_dir, "metrics")
    os.makedirs(system_log_dir, exist_ok=True)
    os.makedirs(metrics_log_dir, exist_ok=True)

    # --- 시스템 로그 설정 (루트 로거 사용) ---
    system_log_file = os.path.join(system_log_dir, f"{exp_name}_system_{timestamp}.log") # <--- 경로 수정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 시스템 파일 핸들러
    system_file_handler = logging.FileHandler(system_log_file)
    system_file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    system_file_handler.setFormatter(system_file_formatter)
    root_logger.addHandler(system_file_handler)

    logging.info(f"System logs will be saved to: {system_log_file}")

    # --- 메트릭 로그 설정 (별도 로거 사용) ---
    metrics_log_file = os.path.join(metrics_log_dir, f"{exp_name}_{timestamp}.jsonl") # <--- 경로 수정
    metrics_logger = logging.getLogger("training_metrics")
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.propagate = False

    # 메트릭 파일 핸들러
    metrics_file_handler = logging.FileHandler(metrics_log_file)
    metrics_file_handler.setFormatter(logging.Formatter('%(message)s'))
    metrics_logger.addHandler(metrics_file_handler)
    
    logging.info(f"Metrics logs will be saved to: {metrics_log_file}")

class MetricsLoggingCallback(TrainerCallback):
    """Logs training metrics using a dedicated logger."""
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            metrics_logger = logging.getLogger("training_metrics")
            log_entry = {"step": state.global_step, **logs}
            try:
                metrics_logger.info(json.dumps(log_entry))
            except Exception as e:
                logging.error(f"Error logging metrics: {e}")
                
                
def parse_args():
    parser = argparse.ArgumentParser(description="SPO Training Script")
    parser.add_argument("--exp_name", type=str, default="spo_math", help="Experiment name for output directory")
    parser.add_argument("--dataset_path", type=str, default="./results/MATH_w_cot_parsed_filter_5_scored_N10_sorted", help="Path to the SPO dataset")
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha parameter for SPO loss")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for SPO loss")
    parser.add_argument("--gamma", type=float, default=0.01, help="Gamma parameter for SPO loss")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps")
    return parser.parse_args()

# --- 4. Main Training Script ---
if __name__ == "__main__":
    # 1. 모델 및 토크나이저 로드
    args = parse_args()
    setup_logging(exp_name=args.exp_name)  # Set up logging with experiment name
    logging.info("Starting SPO training setup...")
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # LoRA를 적용할 모델 로드 (policy_model)
    # torch_dtype=torch.bfloat16 또는 torch.float16을 사용하여 메모리 절약
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2",
                                                 )  # device_map="auto"로 GPU에 자동 할당
    # reference model은 LoRA를 적용하지 않은 원본 모델을 사용합니다.
    ref_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2",
                                                     )  # device_map="auto"로 GPU에 자동 할당
    
    # 패딩 토큰 ID 설정
    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.eval()  # 평가 모드로 전환 (LoRA 적용 시 필요)
    # ref_model.to('cuda')

    # --- LoRA 설정 시작 ---
    # LoRA 하이퍼파라미터 정의
    lora_config = LoraConfig(
        r=8, # LoRA 랭크 (작을수록 메모리 적게 사용, 성능 tradeoff)
        lora_alpha=16, # LoRA 스케일링 팩터
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # LoRA를 적용할 모듈 (LLaMA 계열)
        lora_dropout=0.05, # LoRA 레이어에 적용할 드롭아웃
        bias="none", # 바이어스 파인튜닝 여부 (none, all, lora_only)
        task_type=TaskType.CAUSAL_LM, # 작업 유형 (인과적 언어 모델링)
    )

    # 기본 모델에 LoRA 어댑터 추가 (policy_model에만 적용)
    model = get_peft_model(model, lora_config)
    # 학습 가능한 파라미터 수 확인 (LoRA 적용 후 훨씬 적어짐)
    model.print_trainable_parameters()
    # --- LoRA 설정 끝 ---
    instruction: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: "
    # 데이터셋 인스턴스 생성
    train_dataset = load_from_disk(args.dataset_path)  # 데이터셋 경로에서 로드
    data_collator = SPODataCollator(tokenizer, instruction = instruction, max_length=1024)

    # 3. SPO Loss 함수 인스턴스 생성
    spo_loss_fn = SPOLoss(alpha=args.alpha, beta=args.beta, gamma_score = args.gamma, reference_model=ref_model)
    metrics_callback = MetricsLoggingCallback()
    # 4. TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=f"./results/{args.exp_name}_e{args.num_epochs}",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        # per_device_eval_batch_size=1,    # Example: Set batch size for evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,          # Example: Set learning rate
        warmup_ratio=0.005,                # Example: Number of warmup steps
        report_to='none',         # Report to TensorBoard
        logging_steps=2,
        # eval_strategy="steps",     # Evaluate every `eval_steps`
        # eval_steps=100,                   # Example: Evaluate every 50 steps
        save_strategy="steps",           # Save checkpoint every `save_steps`
        save_steps=20,                   # Example: Save every 50 steps
        remove_unused_columns=False,  # Important for DataCollatorForChatML
        bf16=True,                # Enable mixed precision training
        # Add other arguments like learning_rate, gradient_accumulation_steps etc. as needed
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},  # PyTorch ≥2.1
    )

    # 5. CustomTrainer 인스턴스 생성 및 학습 시작
    trainer = CustomSPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        spo_loss_fn=spo_loss_fn,
        callbacks=[metrics_callback],
    )

    print("Starting SPO training with Best-of-n and Ranked preferences...")
    trainer.train()
    print("SPO training finished!")

    # trainer.save_model("./final_spo_model_with_ranked")
    print("Model saved to ./final_spo_model_with_ranked")