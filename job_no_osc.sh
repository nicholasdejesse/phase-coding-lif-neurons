#!/bin/bash
#SBATCH -J phase_coding_threshold_baseline
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --account=owner-gpu-guest -p notchpeak-gpu-guest
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mail-user u6076585@umail.utah.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:05:00
#SBATCH --gres=gpu:1

module use $HOME/MyModules
module load miniforge3
eval "$(conda shell.bash hook)"

conda activate /uufs/chpc.utah.edu/common/home/u6076585/software/pkg/miniforge3/envs/phase_coding_snn
cd /uufs/chpc.utah.edu/common/home/u6076585/phase-coding-lif-neurons
export PYTHONPATH="$PWD:$PYTHONPATH"
python ./experiments/smnist/smnist_train_alif_tbptt_phase_delta.py --delta-base-threshold 0.2 --delta-wave-amplitude 0.15 --delta-wave-frequency 56 --oscillate-threshold