#!/bin/bash

#SBATCH --job-name=corr            #This is the name of your job
#SBATCH --cpus-per-task=22             #This is the number of cores reserved
#SBATCH --mem-per-cpu=2G              #This is the memory reserved per core.


#SBATCH --time=01:00:00         #This is the time that your task will run 01:00:00 or 1-00:00:00
#SBATCH --qos=6hours	        #You will run in this queue 6hours or 1day or 1 week
#SBATCH --array=1-32           #This is an array job with 200 tasks with a maximum simultaneous number of 20 tasks: 1-200

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=outputs/animal_session%A_%a.o     #These are the STDOUT and STDERR files #j for jobID
#SBATCH --error=outputs/animal_session%A_%a.e
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user="sergej.maul@unibas.ch"        #You will be notified via email when your task ends or fails

#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


source /scicore/home/donafl00/GROUP/Anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate animal_sergej
#pip install -r /scicore/home/donafl00/mauser00/code/AnimalClass/requirements.txt
#git clone https://github.com/donatolab/manifolds.git
#python /scicore/home/donafl00/mauser00/code/AnimalClass/Scicore/create_commands_list.py DON-009191

# Run the corresponding commands from the file commands.cmd one by one
#python create_corr.py
$(head -$SLURM_ARRAY_TASK_ID commands_create_corr.cmd | tail -1) 