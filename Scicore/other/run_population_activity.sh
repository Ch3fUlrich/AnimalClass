#!/bin/bash

#SBATCH --job-name=pop_act              #This is the name of your job
#SBATCH --cpus-per-task=128             #This is the number of cores reserved
#SBATCH --mem-per-cpu=2G                #This is the memory reserved per core.

#SBATCH --time=00:15:00                 #4-00:00:00    #This is the time that your task will run 01:00:00 or 1-00:00:00
#SBATCH --qos=30min                     #1week      #You will run in this queue 30min or 6hours or 1day  or 1week
#SBATCH --array=1-389                   #1-200%20: This is an array job with 200 tasks with a maximum simultaneous number of 20 tasks

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=outputs/intrinsic_pop_act_%j_%A_%a.o       #These are the STDOUT and STDERR files #j for jobID
#SBATCH --error=outputs/intrinsic_pop_act_%j_%A_%a.e
#SBATCH --mail-type=END,FAIL,TIME_LIMIT                 #BEGIN, END,FAIL,TIME_LIMIT
#SBATCH --mail-user="sergej.maul@unibas.ch"             #You will be notified via email when your task ends or fails

#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


source /scicore/home/donafl00/GROUP/Anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate imgartdet
#pip install -r /scicore/home/donafl00/mauser00/code/AnimalClass/requirements.txt
#git clone https://github.com/donatolab/manifolds.git
#python /scicore/home/donafl00/mauser00/code/AnimalClass/Scicore/create_commands_list.py DON-009191

# Run the corresponding commands from the file commands.cmd one by one
# File should be filled with python /scicore/home/donafl00/mauser00/code/AnimalClass/Scicore/AnimalClass_command_line.py animal_id session_id
$(head -$SLURM_ARRAY_TASK_ID commands_population_activity.cmd | tail -1) 
#python /scicore/home/donafl00/mauser00/code/Imaging-Artefact-Detector/Scicore/other/population_activity.py DON-014847 20230126