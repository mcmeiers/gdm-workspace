#!/bin/bash
#SBATCH --mail-user=mcmeiers@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH --qos=regular
#SBATCH --time=48:00:00
#SBATCH --account=mp107
#SBATCH --nodes=3
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --constraint=haswell
#SBATCH --output=./logs/gdm1+5+2-fixed-%j.out
#SBATCH --error=./logs/gdm1+5+2-fixed-%j.err
#SBATCH --image=docker:mcmeiers/gdm:latest
#SBATCH --volume="/global/cscratch1/sd/mcmeiers:/opt/project"

#export OMP_PROC_BIND=true
#export OMP_PLACES=threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -n 6 shifter /opt/conda/bin/python MCMC-LCDM-GDM-alpha+5w+2c+fixed_ends.py
