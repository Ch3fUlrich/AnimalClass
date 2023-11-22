
# Init Environment
1. clone AnimalClass git repository 
2. go into AnimalClass directory
3. clone manifolds git repository 
4. go into Scicore directory
5. open and eddit create_commands_list.py
    a. change root_dir to your project root directory: eg. ```root_dir = "/scicore/projects/donafl00-calcium/Users/Sergej/Steffen_Experiments"```

This have do be done only 1 time:
```bash
# 1. clone AnimalClass git repository 
git clone https://github.com/Ch3fUlrich/AnimalClass.git
# 2. go into AnimalClass directory
cd AnimalClass
# 3. clone manifolds git repository 
git clone https://github.com/Ch3fUlrich/manifolds.git
# 4. go into Scicore directory
cd Scicore
# 5. open and eddit create_commands_list.py
# if you want to eddit files in the terminal: vim create_commands_list.py
# create_commands_list.py #root_dir set correctly
```
# Run Scicore 
1. Define Animals and Sessions to be used: Only the Session will be used where Animal-ID and Sesssion-Date match
2. conda activate animal_sergej
3. create commands for sbatch script: Data is located in Project folder. Examples:
4. run sbatch script
This have do be done every time you want to run new sessions:
```bash
#1. Define Animals and Sessions to be used: Only the Session will be used where Animal-ID and Sesssion-Date match
# writing "all" or "" will run all animals/sessions
ANIMALS="DON-009191 DON-009192" 
SESSIONS="20220303 20220304"
# 2. conda activate animal_sergej
conda activate animal_sergej
# 3. create commands for sbatch script
python create_commands_list.py ANIMALS SESSIONS #alls session from DON-009191
# 4. run sbatch script
sbatch run_pipeline.sh
```
