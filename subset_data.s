#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=18:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=dsga3001-subset_data
#SBATCH --mail-type=END
#SBATCH --mail-user=as12453@nyu.edu
#SBATCH --output=slurm_subset_%j.out
  
# Refer to https://sites.google.com/a/nyu.edu/nyu-hpc/documentation/prince/batch/submitting-jobs-with-sbatch

# for more information about the above options

# Remove all unused system modules
module purge

# Move into the directory that contains our code
SRCDIR=$HOME/Yelp-Recommendation-System

# Activate the conda environment
source ~/.bashrc
conda activate dsga3001

# Execute the script
python ./yelp_restaurant_subset.py

# And we're done!
