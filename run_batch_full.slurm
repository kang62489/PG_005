#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH -J PG005_batch
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@oist.jp

# Description: Batch process all tiff/abf pairs with GPU acceleration on Saion
# Expected runtime: ~4-6 hours with GPU (Tesla P100)
# Memory: 32GB (image processing with large TIFF stacks)
# CPUs: 16 cores (for parallel preprocessing)
# GPU: 1x Tesla P100

echo "=========================================="
echo "Job started on $(date)"
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
else
    echo "No GPU detected - will use CPU mode"
    echo ""
fi

# Navigate to project directory
cd $HOME/PG_005 || exit 1

# Activate virtual environment
# NOTE: Create .venv on login node first! (See SETUP_CLUSTER.md)
source .venv/bin/activate

# Verify Python environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run batch processing
echo "Starting batch processing..."
export QT_QPA_PLATFORM=offscreen
python batch_process.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Batch processing completed successfully!"
    echo "=========================================="

    # Copy results to bucket
    echo ""
    echo "Copying results to bucket..."
    bash copy_results_to_bucket.sh

    if [ $? -eq 0 ]; then
        echo "✓ Results copied to bucket successfully!"

        # Display summary from bucket database
        echo ""
        echo "Results saved to: /bucket/WickensU/Kang/datasets/results/results.db"
        python -c "
import sqlite3
conn = sqlite3.connect('/bucket/WickensU/Kang/datasets/results/results.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM experiments')
print(f'Total experiments in database: {cursor.fetchone()[0]}')
cursor.execute('SELECT exp_date, COUNT(*) as n FROM experiments GROUP BY exp_date ORDER BY exp_date')
print('\nBreakdown by date:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} pairs')
conn.close()
"
    else
        echo "⚠ Warning: Failed to copy some results to bucket"
    fi

    echo ""
    echo "Job finished on $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Batch processing failed!"
    echo "Job finished on $(date)"
    echo "=========================================="
    exit 1
fi
