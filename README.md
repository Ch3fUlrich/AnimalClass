# AnimalClass
Tool for working with brain imaging datasets.

## Run MESC to TIFF to Suite2P + BINARIZATION on Scicore
#TODO: continue creating an easier code
### Init Environment
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
### Run Scicore 
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


## General Workflow
### Folder Structure
```
───DON-019608
    │   DON-019608.yaml
    │   
    ├───20240126
    │   │   20240126.yaml
    │   │   
    │   ├───002P-F
    │   │   │     DON-019608_20240126_002P-F_S1-S2-ACQ.mesc
    │   │   │   
    │   │   └───plane0
    │   │           F.npy
    │   │           Fneu.npy
    │   │           iscell.npy
    │   │           ops.npy
    │   │           spks.npy
    │   │           stat.npy
    │   └───TRD-2P
    │            DON-019608_202401261_TRD-2P_S1-ACQ.mat
    │
    └───20240127
        │   ...
       ...
```
### Create Yaml Files
It is possible to manually add properties if you create an ```animal_summary.yaml``` file and edit those properties in it. Those properties will override properties from the Excell Sheet.
**This is espically usefull if Munits in a MESC file do no match the description of Sessions in the MESC file name** (e.g. MUnits: 0,1,2,3 and Naming: S1,S2)
```yaml
DON-014838:
  sessions:
    '20230227':
      UseMUnits:
      - - DON-014838_20230227_002P-F_S1-S2-ACQ.mesc
        - - 0
          - 2
```
#### Example Yaml Files
##### Example Animal Yaml File
```yaml
animal_id: DON-002865
cohort_year: 2021
dob: '20200625'
sex: male
```
##### Example Session Yaml File
```yaml
date: '20210211'
method: 2P
```
#### Convert Excell sheet to yaml files
- Run ```Tools/yaml/Make_yaml.ipynb```
or
- Run command line ```Tools/yaml/yaml_creator.py``` inside the folder within [this folder structur](#Yaml Creation Folder Structures)
##### Yaml Creation Folder Structures
###### Only on MESC Files in a Folder ( auto-folder creation)
```
───Animals
        DON-019608_20240126_002P-F_S1-S2-ACQ.mesc 
        DON-019608_20240126_002P-F_S1-S2-ACQ.mesc
        ...
        yaml_creator.py
        Intrinsic_CA3_database-September_7,_10_08_AM.xlsx
        animal_summary.yaml
```
###### Already present Folder Structure
```
───DON-019608
    │   
    ├───20240126
    │   │   
    │   └───002P-F
    │            DON-019608_20240126_002P-F_S1-S2-ACQ.mesc
    └───20240127
        │   ...
       ...
```
### Run Pipeline
Run Notebook ```OwnClass.ipynb```
#### Summary
```python
# Load Data
animals = load_all(root_dir, wanted_animal_ids=["all"], wanted_session_ids=["all"])
animal_id = "DON-019608"
session_id = "20240126"
session = animals[animal_id].sessions[session_id]

# Generate TIFF from MESC
self.generate_tiff_from_mesc(generate=True, regenerate=False)

# Run Suite2P
self.generate_suite2p(generate=True, regenerate=False)

# Merge MUnits
units = session.get_units(restore=True, get_geldrying=True, unit_type="single", generate=True, regenerate=False)
merged_unit = session.merge_units(generate=True, regenerate=False)
#merged_unit = session.merge_units(generate=True, regenerate=False, compute_corrs=True, delete_used_subsessions=False)

# Binarize
# Create Pairwise Correlations + Correlation Matrix
session.generate_cabincorr(generate=True, regenerate=False, unit_ids="all", compute_corrs=True)
session.load_corr_matrix(generate=True, regenerate=False, unit_id="merged")

# Convert Movement Data (from matlab to python)
# create movement data from data.mat  files in TRD-2P folders
session.convert_movement_data()

# Merge Movement Data and Load
session.merge_movements()
wheel, triggers, velocity = session.load_movements(merged=True, regenerate=False, movement_data_types=["wheel", "triggers", "velocity"])
```

#### Create Plots
##### Statistic Prints/Plots
```python
cell_numbers_dict = extract_cell_numbers(animals)
# Create table to show statistics for comparison of S2P vs Own Pipeline
pipeline_stats = summary_df_s2p_vs_geldrying(cell_numbers_dict)
display(pipeline_stats)

# Show all datasets below 200 cells
print("-----------------------Show all datasets below 200 cells------------------------------------------")
iscellsum = 0
for animal_id, sessions in cell_numbers_dict.items():
    for session_id, session in sessions.items():
        if session_id == "pdays":
            continue
        if (session["iscell"])<200 and (session["iscell"])>0:
            print(f"{animal_id} {session_id} {session["iscell"]}")
        if not session["gel_corr"]:
            print(f"{animal_id} {session_id}: missing geldrying correlation")

viz = Vizualizer(animals, save_dir=Animal.root_dir)
# Plot to show survived cell percentages
viz.show_survived_cell_percentage(pipeline_stats=pipeline_stats)

# Plot number of usefull sessions counted by pday
viz.show_session_pday_histo(cell_numbers_dict=cell_numbers_dict, min_num_cells=200)

# Create table to show statistics for comparison of S2P vs Own Pipeline
# Plot number of cells comparisson S2P vs Geldrying cleaned S2P
viz.show_survived_cell_numbers(cell_numbers_dict=cell_numbers_dict, min_num_cells=200)
viz.show_usefull_sessions_comparisson(cell_numbers_dict=cell_numbers_dict, min_num_cells=200)


#show tabular visualization of usefull mice
pday_cell_count_df = viz.plot_usefull_session_pdays()
pday_cell_count_df = viz.plot_usefull_session_pdays(different_color_after_k_green=15)
```

##### Dataset/Session Plots
```python
Animal.root_dir = root_dir
viz = Vizualizer(animals, save_dir=Animal.root_dir)
unit_ids = ["all", "merged"]
unit_id = "merged"
session = animals[animal_id].sessions[session_id]

# Plot Bursts
fluorescence_types = ["DFF", "F_detrended"]
for fluorescence_type in fluorescence_types:
  viz.bursts(animal_id, session_id, fluorescence_type=fluorescence_type, unit_id=unit_id, remove_geldrying=True)

# Plot Histogram
corr_matrix, pval_matrix = viz.pearson_hist(animal_id, session_id, unit_id=unit_id, 
                                                                remove_geldrying=True, generate_corr=False,
                                                                color_classify=True)

# Plot KDE
## for every mouse (Kernel Density Estimation)
filters = list(animals.keys())
x_axes_ranges = [[-0.1, 0.3], [-0.2, 0.5], [-0.75, 1.05]]
for x_axes_range in x_axes_ranges:
    for unit_id in unit_ids:
            for filter in filters:
                viz.pearson_kde(filters=filter, x_axes_range=x_axes_range, unit_id=unit_id, remove_geldrying=True)

## for every year averaged by age
filters = [2021, 2022, 2023, []] # summarize every year, averaged by pday; [] summarize all years, averaged by pday
x_axes_ranges = [[-0.1, 0.3], [-0.05, 0.1], [-0.02, 0.02]]
for x_axes_range in x_axes_ranges:
    for unit_id in unit_ids:
            for filter in filters:
                viz.pearson_kde(filters=filter, x_axes_range=x_axes_range, unit_id=unit_id, remove_geldrying=True, average_by_pday=True)

# Plot Mean and Standardeviations of Traces
filters = [2021, 2022, 2023, []] # [] summarize all years, averaged by pday
for unit_id in unit_ids:
    for filter in filters:
          viz.plot_means_stds(filters=filter, unit_id=unit_id, remove_geldrying=True)
```

##### Cell propertie Plots
```python
#
Animal.root_dir = root_dir
viz = Vizualizer(animals, save_dir=Animal.root_dir)
unit_ids = ["all", "merged"]
unit_id = "merged"
session = animals[animal_id].sessions[session_id]

merged_unit = session.merge_units(generate=False, regenerate=False, compute_corrs=False, delete_used_subsessions=False)
units = session.get_units(restore=False, get_geldrying=True, unit_type="summary", generate=False, regenerate=False)
print(f"-----------------------------------Plotting Full Session fottprints, contours(standard, merged)----------------------------------------")
for unit_id, unit in units.items():
    viz.unit_footprints(unit)
    viz.unit_contours(unit, remove_geldrying=True)
print(f"-----------------------------------Plotting Unit Contours combinations-----------------------------------")
# print contours of all combination of units size 2
##################################S2P Registration (Footprint position shift determination)##############################
best_unit = session.get_most_good_cell_unit()
min_num_usefull_cells = best_unit.num_not_geldrying() / 3
units = session.get_usefull_units(min_num_usefull_cells)
for unit_id, unit in units.items():
    if unit_id != best_unit.unit_id:
        if unit.yx_shift == [0, 0]:
            session.calc_unit_yx_shifts(best_unit, units)
            break
from itertools import permutations
unit_ids = list(units.keys())
combinations = list(permutations(unit_ids, 2))
#Plotting shifted contours
for combination in combinations:
    if combination[0] < combination[1]:
        plt.figure(figsize=(20, 20))
        viz.multi_unit_contours(units, combination=combination, plot_center=True, shift=True)
print(f"-----------------------------------Plotting Merged Session fluoresence classification-----------------------------------")       
viz.unit_fluorescence_good_bad(merged_unit, batch_size="all", interactive=False, plot_duplicates=False)
```

### Extract Files 
- Run ```Tools/Extractor/Animal_Extractor.ipynb```
Extracted data is saved in extracted folder in ```root_dir```
```python
# Create extractor class
ex = information_extractor(root_dir, wanted_animal_ids=wanted_animal_ids, print_loading=False)

# Extract cabincorr output (root_dir/Animal_ID/Session_ID/002P-F/tif/suite2p_merged/plane0/binarized_traces.npz)
ex.cabincorr(folder_name_content="merged")

# Extract part of cabincorr output (root_dir/Animal_ID/Session_ID/002P-F/tif/suite2p_merged/plane0/binarized_traces.npz)
ex.from_cabincorr(unit_id="merged", data_types=["F_detrended", "F_upphase"])

# Extract fps of Femtonics recourding from MESC file (root_dir/Animal_ID/Session_ID/002P-F/Animal_ID_Session_ID_002P-F_S1-ACQ.mesc)
ex.fps_mesc()

# Extract suite2p output files from suite2p folder (root_dir/Animal_ID/Session_ID/002P-F/tif/suite2p_merged/plane0)
ex.from_suite2p_folder(file_names=["cell_drying.npy", "stat.npy", "cell_drying.npy"], override=False)
ex.from_suite2p_folder(file_names=["allcell_clean_corr_pval_zscore.npy", "cell_drying.npy"])
ex.from_suite2p_folder(file_names=["cell_drying.npy"])

# Extract movement files from TRD-2P folder (root_dir/Animal_ID/Session_ID/TRD-2P)
ex.from_movement(movement_types="velocity", override=False)

# Extract yaml files from animal and session folders (root_dir/Animal_ID/ and root_dir/Animal_ID/Session_ID/)
ex.yaml(override=False)
```
