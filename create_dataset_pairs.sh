#!/bin/bash

#SBATCH --job-name=create_pairs_wav
#SBATCH --partition=compute
#SBATCH --account=your_account_here
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/dataset_pairs.out
#SBATCH --error=logs/dataset_pairs.err

# Create Simple Pairs Dataset (WAV Format) on CPU
echo "=================================================="
echo "Creating Simple Pairs Dataset (WAV Format)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: 16"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module purge
module load python3/2022.01-gcc-11.2.0 # Adjust as needed

# Navigate to project directory
cd your_project_directory_here || { echo "ERROR: Cannot cd to project dir"; exit 1; }

# Create logs directory
mkdir -p logs

# Activate environment
source /path/to/your/venv/bin/activate
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Working directory: $(pwd)"
echo ""

# Run dataset creation
echo "Starting pairs dataset creation..."
echo "Configuration:"
echo "  - Segment duration: 16 seconds"
echo "  - Sample rate: 24000 Hz"
echo "  - Format: WAV (no encoding)"
echo "  - Pairs: consecutive segments (input, output)"
echo "  - Max pairs: 2000"
echo ""
echo "This will process music files and may take 1-2 hours..."
echo ""

python create_dataset_pairs_wav.py

echo ""
echo "=================================================="
echo "Pairs dataset creation completed!"
echo "End time: $(date)"
echo "=================================================="

# Show dataset info
if [ -d dataset_pairs_wav/train ] && [ -d dataset_pairs_wav/val ]; then
    echo ""
    echo "Dataset files created:"
    ls -lh dataset_pairs_wav/
    echo ""
    echo "Train pairs:"
    ls dataset_pairs_wav/train/ | wc -l
    echo "Val pairs:"
    ls dataset_pairs_wav/val/ | wc -l
    echo ""
    echo "Sample files:"
    ls -lh dataset_pairs_wav/train/ | head -6
    echo ""
    if [ -f dataset_pairs_wav/metadata.txt ]; then
        echo "Metadata:"
        cat dataset_pairs_wav/metadata.txt
    fi
    echo ""
    echo "Ready for training!"
else
    echo ""
    echo "ERROR: Dataset folders not found!"
    echo "Check logs/dataset_pairs.err for errors"
fi
