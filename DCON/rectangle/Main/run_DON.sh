#!/bin/bash
#SBATCH --job-name=DCON-rectangle
#SBATCH --time=0:50:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --output=output-%j.log
#SBATCH --error=error-%j.log
#SBATCH --account=def-dengc
#SBATCH --gpus-per-node=h100:1

# 加载指定版本环境
module load cudacore/.12.2.2
module load python/3.12

echo "---------------- Environment Info ----------------"
echo "CUDA Version: $(nvcc --version)"
echo "Python Version: $(python --version)"
echo "GPU Device: $(nvidia-smi -L | head -n1)"

# 激活虚拟环境
source /scratch/wuwen123/xin/DCON/bin/activate

# 运行代码
python3 /scratch/wuwen123/xin/TL-DCON/DCON/rectangle/Main/exp_pinn_plate.py

# 停用虚拟环境（可选）
deactivate