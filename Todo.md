# <ins>**Todos ordered by importants**</ins>
- [x] Create cleaned vs old Dataset
  - [ ] all pictures as bevore
  - [ ] Show # good cells/session/mouse for every session
    - [ ] Min # of bursts
    - [ ] run at Scicore
      - [ ] 2021
      - [ ] 2022
      - [ ] 2023
    - [ ] Plot
      - [ ] red line at min # of cells = 100 
- [ ] Create Picture HTML for explanation of classes, methods, plots
- [ ] Intrinsic Imaging Pipeline
    - [x] Write Animal class 
    - [x] Session
    - [x] Vizualizer
    - [x] Fixer
    - [x] Analyzer
    - [ ] Cell
  - [ ] Code Algo Autodetection for Errors in Data 
    - [ ] Error types
      - [x] autodetect error if pearson correlation histogram is not normally distr around 0. (max diff 0.3)
        - [x] **Create plot**
      - [x] **increasing bursts at the end** Gel drying
      - [ ] **Offset at Start**
      - [ ] **Offset changes**
      - [ ] **Sudden bubble** ask Steffen for more information
      - [x] **Correlations match**? with steffens spreadsheet
    - [x] Get spreadsheet of the intrinsic data is for 2021 and 2022 **Intrinsic_CA3_database-September_7,_10_08_AM**
  - [ ] **replicate problem**
    - [x] Builder correlation graph pictures shown in slack chat with sergej
    - [x] At first by person correlation, afterwards histogram with pearson for
    - [x] Correlations
    - [ ] Density Graphs

- [ ] Write Paper

- [ ] PR about work
  - [ ] How many bad Sessions in %?
  - [x] Are they detected by Steffen?

- [ ] update binarization code for Nathalies 

- [ ] Get to know
  - [ ] Paper from Catalin
    - [ ] Overview
    - [ ] Detailed
  - [ ] Paper from Andres
    - [x] Talk to Andres, so he can explain the setting
    - [ ] Powerpoint Summarys
    - [ ] Detailed
    - [ ] PR from 09.05.23 was send in slack 
  - [ ] 2PMicroscopy
      - [x] get familiar with 
          - [x] experiments especially the [ca] imaging ones 
          - [x] the methods 
              - proprocessing --> Suite2P (https://github.com/MouseLand/suite2p)
      - [ ] Problems
          - [ ] PMT is noisy because random photons
          - [ ] Baseline is not equally set if session is bad 
  
##  Own Questsions
  - [ ] Connectomex --> Neurons + connections (do they have usefull data forstatistics)

## <ins>**Package CaBinCorr Paper (until end of summer)**</ins>
### **Todos**
- [ ] Ask Lukas how to programm the best way or big programm
  - [x] Which API to provide? Should be easy
  - [ ] Unit tests for every funciton
    - [ ] short functions
  - [ ] Design Pattern / Methods:
    - [ ] Object-oriented &rarr; clean code
    - [x] SCRUM / Agiles Developmen
    - [x] SOLID
      - [ ] Single Responsibility,  Open-Closed, Liskov Substitution,  Interface Segregation, Dependency Inversion 
    - [x] DRY (DON’T REPEAT YOURSELF)
    - [x] KISS (KEEP IT SIMPLE STUPID)
    - [x] YAGNI (You aren't gonna need it) &rarr; do not write code which is not needed until now

- [ ] Create clean Git Repo
  - [ ] Create Parameter filed
  - [ ] Output:
    - [ ] Figures
    - [ ] Optional wheel movement data
    - [ ] Data
      - [ ] BinTraces.npz
      - [ ] correlations
      - [ ] Duplicates
  - [ ] calcium object wrapper for suite2p
  - [ ] binarization
  - [ ] correlation --> deduplication
- [ ] freeze packages
  - [ ] put wrapper around
  - [ ] create pypi (organization which created pip) (edited) 
- [ ] Write Paper

## <ins>**Detailed Information for Intrisic Imaging Pipeline**</ins>
### **Todos**
- [ ] Create Class 
  - [ ] Detector?
    - [ ] run s2p for 
  - [x] Visualizer
    - [x] Traces
    - [x] Raster
    - [x] Histograms
    - [x] KDE
  - [ ] Analyzer
  - [x] Animal
    - [x] Session
      - [x] Run Cabincorr
      - [ ] Cell
        - [ ] Save .npz
          - [ ] Parameter
            - [ ] CellID, 
            - [ ] IsCell, 
            - [ ] Duplicate, 
            - [ ] S2P Flag, 
            - [ ] Burstrate, 
            - [ ] df/F, 
            - [ ] Contour
  - [ ] Rodrigo should be able to use it for Graph visualization
  
- [ ] **Code Algo Autodetection** for Errors in Data + Solutions
  - [x] Show Flavio amount of errors
    - [x] autodetect error if pearson correlation histogram is not normally distr around 0. (max diff 0.3)
      - [x] **Create plot** for every mouse/year
        - [x] mean/std on y-axis and 
          - [ ] which mean, std? 
        - [x] the day of recording for mouse on x-axis.
        - [x] include steffens colors
        - [x] Validate detection by comparing with 2021
        - [x] Create KDE
  - [ ] Detector Class
  - [ ] Fixer Class
  - [ ] 1. **Preparation**
    - [x] Take Bad Session (9191 20220227, 9192 20220221, 20220319)
    - [x] break up in parts
    - [x] run suite2p on every part  of a session (should solve baselineshift)
      - [x] Use Box filter (low pass: 0.5Hz cutoff, 30Hz sample rate, 2 degree) instead of traces 
      - [x] Look in s2p gui: look okay?<span style="color:green">Unit 1, 3, 4 not</span> many bad cells?<span style="color:green">prob. at unit 4</span> gel artifact?<span style="color:green">yes</span>
      - [ ] Create Metrics for Units
        - [x] rasters plots
        - [x] human good/bad evaluation (bad: unit 1, 3, 4 good: unit 2)
        - [x] Autodetection Geldrying
          - [x] MultiUnitActivityHistogram (MUAH)
            - [x] Bin Raster plots in 1sec. or 1Frame, Sum all spikes
            - [x] can we detect really bad sessions on Rasterplots? --> NO
              - [x] Linear regression for slope detection --> BAD 
              - [x] Detect for every cell if cell is good or bad
                - [x] sliding mean 
                - [x] output saved in unit.cell_drying and cell_drying.npy
        - [ ] create yaml file for session data
    - [ ] Datastructures to work with

      - [x] S2P cell footprints/contours
        - [x] merge
          - [x] c.load_footprints() ..........................................................
            - [x] self.contours = contours  #outline of the cells
            - [x] self.footprints = imgs    #boolean mask for the cells
            - [x] use footprint and make boolean footbrints --> multiply with image 

        - [x] deduplicate &rarr; unique mask
          - [x] remove bad cells by running [deduplication notebook](https://github.com/donatolab/manifolds/blob/main/donlabtools/correlation/Deduplicate_neurons.ipynb)
            - [x] on parts of sessions for testing
            - [x] on concatenated data
            - [x] look at deleted cells in deduplication.png 
          - [x] Create Visualization and figure to show all cell contours
            - [x] keep track of deleted cells and contours
            - [ ] move to figure directory
      - [x] Create movement corrected data with s2p using a flag!!!!!!
      - [x] RAW ca data
        - [x] access hopefully motion corrected raw data (S2P has a settable flag)
        - [x] generate Traces
          - [x] motion corrected data could have shifts between each unit
            - [x] possible to align with s2p? <span style="color:green">yes</span>
        - [x] Run data and own mask in S2P if possible 
            - [x] S2P cell traces (trace = sum of area)
            - [x] load and binaryze at the end
              - [x] np.memmap() #memory map instant data load 
      

  - [ ] Network **density** way to high
    - [ ] **Offset at Start** &rarr; Pearson correlation > **0.7** 
      - [ ] <span style="color:green">Normalize with Common Average Referencing (CAR)</span>
    - [ ] Is it really an error?
  - [ ] High amount of **increasing bursts at the end**
    - [ ] Are these even neurons?
    - [ ] Gel dries out ?? &rarr; <span style="color:green">Cut at the end</span>
  - [ ] **Baseline changes** in burst analysis
    - [ ] Detect by sliding window mode of distribution
      - [ ] https://centre-borelli.github.io/ruptures-docs/
      - [ ] Hartigan Dip Test
    - [ ] Because interrupts and new starts &rarr; new initiaion ????
      - [x] <span style="color:green">Run Suite2P on parts of Session &rarr; multiple masks</span>
      - [ ] <span style="color:green">merge masks analyse whole Session &rarr; combine mescs &rarr; Unique Neuron activity</span>
  - [x] **Sudden bubble**Strange behaviour of activity (not sure if only at the end)
  - [x] **Correlations match** with steffens spreadsheet NO
  - [ ] Compare Detections with Excel Sheet with detailed info about occured problems of Steffen (ask where located)
    - [x] Add additonal row to excel sheet for autodetected errors 

- [ ] Do Rodrigos Job
  - [ ] Graph Analysis on good Data after Bad Data was filtered out 
    - [ ] Bad Sessions really bad or realistic?
- [ ] Be able to create own versions of Binarization and Person correlated data
  - [ ] Decide which is better for our goal (Build networks based on activity)

### Done
- [x] Initial tests
  - [x] _.mesc_ (1,2,3) to .tiff with [Mesc to Tiff] 
  - [x] run [Suit2p] on the raw .tiff file with sampling rate to 30 fps --> 1.5h 
- [x] create beautiful code to proof flavio wrong in bad mice data deletions
    - [x] overlapp KDE
    - [x] overlapp line plots
- [ ] Data Errors



### **Data**
- [x] Connect to Drives
  - [x] [Biozentrum](smb://unibasel.ads.unibas.ch/bz/)
  - [x] [Scicore Biozentrum](smb://toucan-all.scicore.unibas.ch/donafl00-calcium$/) for <ins>Rodrigos data</ins>
  - [x] [Biopz Jumbo](smb://biopz-jumbo.storage.p.unibas.ch/RG-FD02$/_Members/mauser00/Desktop)
    - Middle connection, because Biozentrum is not safe?
    - Data from Steffen &rarr; Catalin/Me process &rarr; Rodrigo for Network Analysis (eg. Density)


- [ ] Important Data: Replicate, Detect, Clean
  - [ ] Focus on Clean Datasets from pubs from 2021, than adults and 2022
    - [ ] 2021
      - [ ] Pups 00608X, 4, 5, 7
      - [ ] Adults ,002865 ,003165 ,003343
    - [ ] 2022
      - [ ] Pubs ,9191, 9192, 10473, 10477
      - [ ] Adults ,2865 ,8497 ,8498 ,8499

- [x] replicate problem
  - [x] Mouse 9192 (most Experiments and most often abd)
    - [x] <span style="color:red">bad</span> Session [20220319] _U:\RG Donato\Microscopy\Steffen\Experiments\DON-009192\20220319_
      - [x] run TOTAL RUNTIME 5205.42s
      - [x] visualized [Notebook for visualize some of the problems] 
    - [x] <span style="color:green">good</span> Session [20220306] (at least somewhat better) _U:\RG Donato\Microscopy\Steffen\Experiments\DON-009192\20220319_
      - [x] run TOTAL RUNTIME 2851.29 sec
      - [x] visualized activity bursts + raster [Notebook for visualize some of the problems]
    - [x] why is my data that bad? &rarr; baseline shifts
      - Because Suite2P used tif Folder with pictures in it (steffen used the pictures for other stuff)


### **Code base**
- Animal Class
- Session Class
- Vizualizer Class
#### OLD
- Own
  - [x] run_function_on_every_dataset(animal_ids_dict, function, catch_errors=True)
    - [x] merge_to_tiff
    - [x] run_suite2p
    - [x] Calcium_binarization_plotsaving
      - [x] Bursts, Rasters
      - [x] Traces
      - [x] Pearson Corr and corresponding coefficient Histogram 
  - [x] Code Steffens Spreadsheet Splitting

- Others
  - [Notebook for visualize some of the problems]
  - [Mesc to Tiff] 


## <ins>**Nathalie Muouse in a Bo**</ins>
- [ ] update binarization method to acutal one
  - Uses csv files
## <ins>**Active Avoidance paradigm**</ins>
- [ ] Try [Cebra](https://cebra.ai/)

## <ins>**Detailed Information for BMI**</ins>
- [ ] BMI GUI
  - [ ] Matplotlib slow &rarr; PyQT? PyQT-Graph? what is faster
- [ ] Analysis needs to be standardized
- [ ] Many utility functions
- [ ] 10 Notebooks
## <ins>**Detailed Information for Graph Stuff**</ins>





[Mesc to Tiff]: https://github.com/donatolab/manifolds/tree/maindonlabtools/renan_tiff_process
[Suit2p]: https://github.com/MouseLand/suite2p
[20220319]:smb://unibasel.ads.unibas.ch/bz/RG%20Donato/Microscopy/Steffen/Experiments/DON-009192/20220319
[Notebook for visualize some of the problems]: https://github.com/donatolab/manifolds/blob/main/donlabtools/intrinsic_dynamics_project/Visualize_suite2p_concatenated_data.ipynb

![pic](../Catalin/Suit2P_test_04.05.2023.jpg)
<img src="../Catalin/Suit2P_test_04.05.2023.jpg" alt="Getting started" />
<img src="https://www.mylifeorganized.net/i/products/notebook.png" style="width: 180px">