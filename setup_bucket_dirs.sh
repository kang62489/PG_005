#!/bin/bash
# Create necessary directories in the bucket
# Run this on the LOGIN NODE, not compute node

echo "=========================================="
echo "Setting up bucket directories"
echo "=========================================="
echo ""

BUCKET_BASE="/bucket/WickensU/Kang/datasets"

echo "Creating directories in: $BUCKET_BASE"
echo ""

# Create main directories
mkdir -p "$BUCKET_BASE/rec_summary"
mkdir -p "$BUCKET_BASE/raw_images"
mkdir -p "$BUCKET_BASE/raw_abfs"
mkdir -p "$BUCKET_BASE/processed_images"
mkdir -p "$BUCKET_BASE/results"

# Check if creation succeeded
if [ -d "$BUCKET_BASE/processed_images" ] && [ -w "$BUCKET_BASE/processed_images" ]; then
    echo "✓ Directories created successfully"
    echo ""
    echo "Directory structure:"
    ls -ld "$BUCKET_BASE"/*
    echo ""

    echo "=========================================="
    echo "Next steps:"
    echo "=========================================="
    echo "1. Upload your data to these directories:"
    echo "   - REC_*.xlsx files → $BUCKET_BASE/rec_summary/"
    echo "   - *.tif files → $BUCKET_BASE/raw_images/"
    echo "   - *.abf files → $BUCKET_BASE/raw_abfs/"
    echo ""
    echo "2. Verify your data:"
    echo "   ls $BUCKET_BASE/rec_summary/*.xlsx"
    echo "   ls $BUCKET_BASE/raw_images/*.tif | head"
    echo "   ls $BUCKET_BASE/raw_abfs/*.abf | head"
    echo ""
    echo "3. Then submit your job:"
    echo "   sbatch run_batch_test_final.slurm"
    echo "=========================================="
else
    echo "✗ ERROR: Could not create writable directories"
    echo ""
    echo "The bucket might be read-only on this node."
    echo "Possible solutions:"
    echo "1. Create directories on login node (not compute node)"
    echo "2. Contact cluster admin about bucket write permissions"
    echo "3. Use alternative path like \$HOME/datasets or /flash/WickensU/Kang/datasets"
    echo ""
    exit 1
fi
