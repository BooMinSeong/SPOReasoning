#!/bin/bash
#SBATCH --job-name=vllm_gen  # Job name

# cpu, gpu Setting
#SBATCH --partition=A6000
#SBATCH --gres=gpu:2                    # Number of GPUs (per node)
#SBATCH --cpus-per-task=3               # CPU cores/threads

# Log Setting
#SBATCH --output=logs/%x-%j.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x-%j.err    # Standard error

# Running Time
#SBATCH --time=12:00:00                  # Time limit hrs:min:sec

# 환경 설정
source ~/.bashrc
conda activate inside

# 모듈 설정
module purge
module load gnu12/12.2.0
module load cuda/12.6

# 컴파일러 명시 설정
export CC=$(which gcc)
export CXX=$(which g++)

# 실행
python vllm_inf.py --start_bin $1


