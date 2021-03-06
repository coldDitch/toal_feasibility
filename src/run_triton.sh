#!/bin/bash -l
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00 --mem-per-cpu=4000
#SBATCH -o ./log/%a.out
#SBATCH --array=0-40

set -xe
module purge
module load anaconda3
module list
which python
env | sort

source activate /scratch/work/kylmaoj1/ana3
srun /scratch/work/kylmaoj1/ana3/bin/python activelearning.py ./ $SLURM_ARRAY_TASK_ID decision_ig 1
srun /scratch/work/kylmaoj1/ana3/bin/python activelearning.py ./ $SLURM_ARRAY_TASK_ID uncer_y 1
srun /scratch/work/kylmaoj1/ana3/bin/python activelearning.py ./ $SLURM_ARRAY_TASK_ID random 1
