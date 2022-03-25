#!/bin/bash
#SBATCH --mail-user=mcmeiers@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH --qos=regular
#SBATCH --time=72:00:00
#SBATCH --account=mp107
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --constraint=haswell
#SBATCH --output=./logs/gdm6-%j.out
#SBATCH --error=./logs/gdm6-%j.err
#SBATCH --image=docker:mcmeiers/gdm:latest
#SBATCH --volume="/global/cscratch1/sd/mcmeiers/gdm6:/opt/project/output"

#export OMP_PROC_BIND=true
#export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -n 4 shifter /opt/conda/bin/python MCMC-LCDM+6GDM-fixed-ends.py
