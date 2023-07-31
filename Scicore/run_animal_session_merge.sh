#!/bin/bash

#SBATCH --job-name="animal_session"                    #This is the name of your job
#SBATCH --cpus-per-task=16                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=15G              #This is the memory reserved per core.
#SBATCH --tmp=50

#SBATCH --time="1-00:00:00"        #This is the time that your task will run
#SBATCH --qos="1day"           #You will run in this queue
#SBATCH --array=1-200%20        #This is an array job with 200 tasks 
  !!!!!!!!!                              #with a maximum simultaneous number of 20 tasks

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output="outputs/animal_session.o"%j     #These are the STDOUT and STDERR files
#SBATCH --error="animal_session.e"%j
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user="sergej.maul@unibas.ch"        #You will be notified via email when your task ends or fails

#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


source /scicore/home/donafl00/GROUP/Anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate animal_sergej
pip install -r /scicore/home/donafl00/mauser00/code/AnimalClass/requirements.txt

conda env export -n suite2p_cat -f job_$SLURM_JOBID_env.yml --no-builds

python /scicore/projects/donafl00-calcium/Users/Sergej/create_commands_list.py animal_id session_id

# Run the corresponding commands from the file commands.cmd one by one
# File should be filled with python /scicore/projects/donafl00-calcium/Users/Sergej/AnimalClass_command_line.py animal_id session_id
!!!!!!!!!!!!!!!!!!!
$(head -$SLURM_ARRAY_TASK_ID commands.cmd | tail -1) 