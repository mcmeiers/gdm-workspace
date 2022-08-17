#!/bin/bash
#SBATCH --mail-user=mcmeiers@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH --qos=regular
#SBATCH --time-min=3:00:00
#SBATCH --time=48:00:00
#SBATCH --account=mp107
#SBATCH --nodes=3
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --constraint=haswell
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --image=docker:mcmeiers/gdm
#SBATCH --volume="/global/cscratch1/sd/mcmeiers:/opt/project"


#SBATCH --comment=96:00:00  #desired timelimit
#SBATCH --signal=B:USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

#export OMP_PROC_BIND=true
#export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -n 6 shifter /opt/conda/bin/python MCMC-LCDM-GDM-1+5+2-resume.py
