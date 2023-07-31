#!/bin/bash

#SBATCH --job-name="suite2p_cat"                    #This is the name of your job
#SBATCH --cpus-per-task=8                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=30G              #This is the memory reserved per core.
#SBATCH --tmp=50

#SBATCH --time="06:00:00"        #This is the time that your task will run
#SBATCH --qos="6hours"           #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output="myrun.o"%j     #These are the STDOUT and STDERR files
#SBATCH --error="myrun.e"%j
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user="mitelutco@gmail.com"        #You will be notified via email when your task ends or fails

#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


source /scicore/home/donafl00/GROUP/Anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate animal_sergej
pip install -r /scicore/home/donafl00/mauser00/code/AnimalClass/requirements.txt

conda env export -n suite2p_cat -f job_$SLURM_JOBID_env.yml --no-builds

python /scicore/projects/donafl00-calcium/Users/Sergej/AnimalClass_command_line.py