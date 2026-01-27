#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0:30:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -J PG005_test
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err

# Test batch processing with GPU acceleration on Saion
# Uses cluster's Python 3.11.4 module

echo "=========================================="
echo "TEST JOB started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load CUDA 11.1 module (compatible with driver 455.32.00)
module load cuda/11.1

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo ""
    echo "Loaded modules:"
    module list
    echo ""
    echo "CUDA Toolkit: $(nvcc --version | grep release)"
    echo "CUDA_HOME: $CUDA_HOME"
    echo ""
else
    echo "No GPU detected - using CPU mode"
    echo ""
fi

# Navigate to project directory
cd $HOME/PG_005 || exit 1

# Activate virtual environment
source .venv/bin/activate

# Verify Python environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo ""

# Test CUDA availability in Python
echo "Testing CUDA in Python..."
python test_cuda.py
echo ""

# Run test batch processing
echo "Starting TEST batch processing..."
export QT_QPA_PLATFORM=offscreen
python test_batch.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "TEST completed successfully!"
    echo "=========================================="

    # Copy results to bucket
    echo ""
    echo "Copying results to bucket..."
    bash copy_results_to_bucket.sh

    if [ $? -eq 0 ]; then
        echo "✓ Results copied to bucket successfully!"
    else
        echo "⚠ Warning: Failed to copy some results to bucket"
    fi

    echo ""
    echo "Job finished on $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: TEST failed!"
    echo "Job finished on $(date)"
    echo "=========================================="
    exit 1
fi
