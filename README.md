# AnimalClass
Tool for working with brain imaging datasets.

## Run MESC to TIFF to Suite2P + BINARIZATION + Pairwise Correlation
### Init Environment in Terminal
1. clone AnimalClass git repository 
2. go into AnimalClass directory
3. clone manifolds git repository 
4. create conda environment and install packages

This has do be done only 1 time:
```bash
# 1. clone AnimalClass git repository 
git clone https://github.com/Ch3fUlrich/AnimalClass.git
# 2. go into AnimalClass directory
cd AnimalClass
# 3. clone manifolds git repository 
git clone https://github.com/Ch3fUlrich/manifolds.git
# 4. create conda environment
conda create -n animal_env python=3.8
pip install -r requirements.txt

```
### Run on Scicore or Locally
#### Jupyter Notebook
1. Open ```Scicore\run_scicore_pipeline_helper.ipynb``` to run on scicor or ```Usage_Example.ipynb``` locally for local usage
2. Set your Environment
3. Run Pipeline

#### Terminal on Scicore
If you run it in Terminal this procedure will be used **MESC-->TIFF-->Suite2p-->Binarize-->Pairwise Correlate**
1. Define Animals and Sessions to be used: Only the Session will be used where Animal-ID and Sesssion-Date match
2. conda activate your_animalclass_environment
3. create commands for sbatch script: Data is located in Project folder. Examples:
4. run sbatch script
This have do be done every time you want to run new sessions:
```bash
#1. Define Animals and Sessions to be used: Only the Session will be used where Animal-ID and Sesssion-Date match
# writing "all" or "" will run all animals/sessions
ANIMALS="DON-009191 DON-009192" 
SESSIONS="20220303 20220304"
# 2. conda activate your_animalclass_environment
conda activate your_animalclass_environment
# 3. create commands for sbatch script
python create_commands_list.py ANIMALS SESSIONS #alls session from DON-009191
# 4. run sbatch script
sbatch run_pipeline.sh
```

#### Usefull Information + Terminal Commands
1. If the jobs stop fast look at the output/error files in the AnimalClass/Scicore/Outputs Folder
2. Check in the Terminal if the jobs are running
``` bash
squeue -u <username> -s
```
1. Check every 60 seconds if the pipeline is running
```bash
   while true;do squeue -u <username> -s; sleep 60; done
```
1. Cancel running job
```bash
    # by Job ID
    scancel <jobID>
    # by username
    scancel -u <username>
```

## General Workflow
### Steps done for Correlation extraction
Extract fluoresence information from a Session Dataset (1 Day recording). Some Steps are not mentioned in this description, since they are not relevant for the correlation extraction.

1. Generate **TIFF** from **MESC**/**RAW** files ```session.generate_tiff```
   1. Using **h5py** package
   2. Create a **RAW** and **TIFF** file for every part (**MUnit**) of the **MESC**

2. Extract **Fluorescence Data** of all parts (**MUnits**) ```session.generate_suite2p```
3. Run **Suite2P** on **TIFF** with default parameters
4. Binarize **Fluorescence Data** and calculate **correlations** ```session.generate_cabincorr```
   1. Run **Catalins Binaraization** and **Correlation** code for every **Unit** given parameters
   2. Merge all **Units** to one **Unit** ```session.merge_units```
   3. Classify **cells** if they are real **cells** or **gel drying artefacts** ```unit.get_geldrying_cells```
      1. Calculate the **mean** and **standard deviation** of **sliding window** (default: 30*60 = 1 minute) **fluorescence** for each **cell**. ```Analyzer.get_all_sliding_cell_stat```
      2. **Geldrying detection** ```Analyzer.geldrying```
      3. Check if the mean of the data increases for 1.5 minutes without a 0.5 minutes break (at 30fps) ```Analyzer.cont_mean_increase```
   4. Determine **MUnit** with most **useful cells** (best **MUnit** without **gel drying artefacts**) ```session.get_most_good_cell_unit```
   5. Determine the **shift** of every **MUnit** to the best **MUnit** ```session.calc_unit_yx_shifts```
      1. Get **reference image** of best **MUnit** ```unit.get_reference_image```
         1. Load first 1000 frames of **raw binary data** ```Binary_loader.load_binary```
            1. Compute **reference image** ```suite2p.registration.register.compute_reference```
         2. Compute **reference mask** ```suite2p.registration.register.compute_reference_masks```
         3. Compute **shift** of every **MUnit** to the best **MUnit** ```unit.calc_yx_shift```
            1. Load first 1000 frames of **raw binary data** ```Binary_loader.load_binary```
            2. Register frames to **reference image** ```suite2p.registration.register.register_frames```
            3. Calculate the **mean shift** in **x** and **y** direction
   6. Merge **stats file** of **MUnits** and **deduplicate**
      1. For all **units**: remove **cells** that are **out of view** after **shift** ```Merger.remove_abroad_cells```
         1. ```Merger.shift_rotate_contour_cloud```
         2. Remove **out of bound cells**
      2. Merge and **deduplicate** **duplicated cells** ```Merger.merge_deduplicate_footprints``` (modified code from **Catalin**)
         1. ```Merger.generate_batch_cell_overlaps```
         2. ```Merger.find_candidate_neurons_overlaps```
         3. ```Merger.make_correlated_neuron_graph```
         4. ```Merger.delete_duplicate_cells```
   7. Update all **MUnits** using ```Merger.shift_update_unit_s2p_files```
   8. Merge **Suite2P files** of all **MUnits** ```Merger.merge_s2p_files```
      1. Concatenate **fluorescence** and other **Suite2P files** in **time**
   9. Create a new **Unit** object with all **merged information**
      1.  Create a **Unit** object ```Unit```
      2.  Determine **wrongly detected cells** ```Unit.get_geldrying_cells```
5.  Create **Correlation Matrix** ```session.load_corr_matrix(unit_id="merged")```
    1.  Run **Catalins code** on all **cells** ```session.get_cells(generate=True)```
    2.  Combine to **matrix**
        1.  **corr_matrix**
        2.  **pval_matrix**
        3.  **z_score_matrix**

#### Provided data to people
Here is a list of data that was provided to people. All data is from the **merged Unit** composed of all **MUnits** of a **Session**. All fluoresence data is based on all time frames and not on only stationary or moving frames.

##### Everton
1.	Run the code from Rodrigo for detecting graph properties and
    - created an Excell Table with information about the graph properties,
    - Boxplots as we saw already and
    - Significants Plots based on a permutation test made by Rodrigo.
2.	Run my code and generated the firing rates and coactivity counts (not normalized) for
    - Stationary
    - Moving
    - Stationary+Moving ( was already provided before ) 

##### Cecillia
- Fluorescence Data
  - F_detrended (without up or down trends)
  - F_upphase (only up phases)
- Suite2p files
  - cell_drying (True/False)
  - stat (Suite2p stats file)
  - is_wheel (True/False if the mouse is running on the wheel) 
- Movement Data
  - velocity (**nan** if no data was available or mouse was on the platform)
- FPS 
- yaml files
  - animal.yaml (one value for every unit)
```yaml
animal_id: DON-003343
cohort_year: 2021
dob: '20200826'
sex: male
```
  - session.yaml
```yaml
cam_data:
- true
- true
- true
- true
comment: low SNR; behav 30 min
date: '20231108'
duration:
- 15.0
- 15.0
- 15.0
- 15.0
expt_pipeline: intrinsic CA3
fucntional_channel: 1
laser_power: 14.0
lens: 16x
light: 920.0
method: 2P
movement_data:
- true
- true
- false
- false
n_channel: 1
n_planes: 1
pixels: 512x512
pockel_cell_bias: 1650.0
session_parts:
- S1
- S2
- S3
- S4
setup: femtonics
ug_gain: 74.0
underground:
- wheel
- wheel
- platform
- platform
ur_gain: null
weight: 14.1
```

##### Rodrigo
- Fluorescence Data
  - F_upphase (only up phases)
- Suite2p files
  - allcell_clean_corr_pval_zscore (correlation, p-value, z-score of all cells)
  - cell_drying (True/False if the cell is a gel drying artefact)
- movement data
  - velocity (**nan** if no data was available or mouse was on the platform)
- FPS
- yaml files
  - animal.yaml
  - session.yaml

#### Parameters
##### Suite2p
Table of not default [Suite2P Parameters](https://suite2p.readthedocs.io/en/latest/settings.html)
| Parameter | Description | Value |
| --- | --- | --- |
| fs | frame rate of the movie | **30** |

##### Catalins Binaraization and Correlation
Table of used parameters
| Parameter | Description | Value |
| --- | --- | --- |
| parallel_flag | parallel | **True** |
| animal_id | animal_id | **DON-XXXXXX** |
| recompute_binarization | recomputes binarization of input data | **False** |
| remove_ends | delete the first and last x seconds in case [ca] imaging had issues | **False** |
| detrend_filter_threshold | this is a very low filter value that is applied to remove bleaching before computing mode|  **0.001** |
| mode_window | None: compute mode on entire time; Value: sliding window based - baseline detection # of frames to use to compute mode | **30 * 30** |
| dff_min | set the minimum dff value to be considered an event; required for weird negative dff values that sometimes come out of inscopix data | **0.02** |
| data_type = "2p" | define 1p or 2p data structure | **"2p"** |
| remove_bad_cells | removes cells not passing suite2ps criteria | **True** | 
| detrend_model_order | ???????????? | **1** |
| percentile_threshold | ???????????? | **0.000001** |
| | | |
| | | |
| corr_parallel_flag | to run correlations in parallel or not | **True** |
| recompute_correlation | recomputes correlation of input data | **False** |
| binning_window | binning window in frames | **30** |
| subsample | subsample traces by this factor | **1** |
| scale_by_DFF | scale traces by DFF | **True** |
| shuffle_data | shuffle data | **False** |
| subselect_moving_only | subselect moving only | **False** |
| subselect_quiescent_only | subselect quiescent only | **False** |
| zscore | to generate zscores | **True** |
| n_tests_zscore | number of tests for zscore | **1000** |
| n_cores | number of cores to use | **32** |

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
    │   │   └───tif
    │   │       │      
    │   │       └───suite2p
    │   │           │      
    │   │           └───plane0
    │   │               F.npy
    │   │               Fneu.npy
    │   │               iscell.npy
    │   │               ops.npy
    │   │               spks.npy
    │   │               stat.npy
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
#### Convert Excell sheet to yaml files (For more experienced users)
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
### Run Pipeline (For more experienced users)
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
