#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=dsga3001-lab2
#SBATCH --mail-type=END
#SBATCH --mail-user=as12453@nyu.edu
#SBATCH --output=slurm_YelpBias_%j.out
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch

# for more information about the above options

# Remove all unused system modules
module purge

# Move into the directory that contains our code
SRCDIR=$HOME/Yelp-Recommnedation-System

# Activate the conda environment
source ~/.bashrc
conda activate dsga3001

# Execute the script
python ./Bias_and_BiasMF_Models.py

# And we're done!
