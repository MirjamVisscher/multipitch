#!/bin/bash
#SBATCH --partition=gpua16
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH -t 10-00:00              # time limit: (D-HH:MM)


start_time=$(date +%s)

# Load environment variables and activate conda environment
source ~/.bashrc || { echo "Failed to source .bashrc"; exit 1; }
conda activate multipitch_gpu || { echo "Failed to activate conda environment"; exit 1; }
export CUDA_VISIBLE_DEVICES=3
# Change to the working directory
cd /storage/scratch/vissc022/multipitch-gpu || { echo "Failed to change directory"; exit 1; }

# Ensure output directory exists
mkdir -p output/predictions || { echo "Failed to create output directory"; exit 1; }

# Run the Python script with specified arguments
python multipitch.py --experiment "$1" --model "$2" --type "$3" --path "$4" || { echo "Python script execution failed"; exit 1; }

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Runtime: $duration seconds"

