#!/bin/bash
# Copy results from temporary working directory to bucket

echo "=========================================="
echo "Copying results to bucket..."
echo "=========================================="

# Get job ID
JOB_ID=${SLURM_JOB_ID}

if [ -z "$JOB_ID" ]; then
    echo "ERROR: SLURM_JOB_ID not set. Run this from within a SLURM job."
    exit 1
fi

# Determine working directory
if [ -d "/flash/WickensU/Kang/tmp_job_${JOB_ID}" ]; then
    WORK_DIR="/flash/WickensU/Kang/tmp_job_${JOB_ID}"
elif [ -d "$HOME/tmp_job_${JOB_ID}" ]; then
    WORK_DIR="$HOME/tmp_job_${JOB_ID}"
else
    echo "ERROR: Working directory not found"
    echo "Looked for: /flash/WickensU/Kang/tmp_job_${JOB_ID}"
    echo "       and: $HOME/tmp_job_${JOB_ID}"
    exit 1
fi

# Bucket destination
BUCKET_DIR="/bucket/WickensU/Kang/datasets"

echo "Working directory: $WORK_DIR"
echo "Bucket directory: $BUCKET_DIR"
echo ""

# Create bucket directories if they don't exist
mkdir -p "$BUCKET_DIR/raw_images"
mkdir -p "$BUCKET_DIR/processed_images"
mkdir -p "$BUCKET_DIR/results"

# Copy raw images
if [ -d "$WORK_DIR/raw_images" ]; then
    echo "Copying raw images..."
    rsync -av --progress "$WORK_DIR/raw_images/" "$BUCKET_DIR/raw_images/"
    echo "✓ Raw images copied"
else
    echo "⚠ No raw_images directory found"
fi

# Copy processed images
if [ -d "$WORK_DIR/processed_images" ]; then
    echo "Copying processed images..."
    rsync -av --progress "$WORK_DIR/processed_images/" "$BUCKET_DIR/processed_images/"
    echo "✓ Processed images copied"
else
    echo "⚠ No processed_images directory found"
fi

# Copy results
if [ -d "$WORK_DIR/results" ]; then
    echo "Copying results..."
    rsync -av --progress "$WORK_DIR/results/" "$BUCKET_DIR/results/"
    echo "✓ Results copied"
else
    echo "⚠ No results directory found"
fi

echo ""
echo "=========================================="
echo "Summary:"
ls -lh "$BUCKET_DIR/results/results.db" 2>/dev/null && echo "✓ Database: $BUCKET_DIR/results/results.db" || echo "⚠ No database found"
echo "Raw images: $(ls $BUCKET_DIR/raw_images/*.tif 2>/dev/null | wc -l) files"
echo "Processed images: $(ls $BUCKET_DIR/processed_images/*.tif 2>/dev/null | wc -l) files"
echo "Result directories: $(ls -d $BUCKET_DIR/results/*/ 2>/dev/null | wc -l)"
echo "=========================================="

# Ask if user wants to cleanup temp directory
echo ""
echo "Temporary working directory: $WORK_DIR"
echo "Size: $(du -sh $WORK_DIR | cut -f1)"
echo ""
echo "You can remove it with:"
echo "  rm -rf $WORK_DIR"
