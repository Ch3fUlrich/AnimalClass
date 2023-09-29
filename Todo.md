# <ins>**Todos ordered by importants**</ins>
- [ ] run at Scicore
  - [ ] 2023

- recreate the yaml file creation

- run pipeline 
  - [x] with 128gb ram Check error ouptut of 3890636_1, 6, 8, 9
    - [ ] check
  - [ ] eleven (DON2865), five (DON3165), and eight (DON3343)
		    .\Users\unprocessed_intrinsic_CA3\ instrinc_imaging.yaml
  	
- [] Analysis manifolds CEBRA

Help
- [ ] Anja
  - [ ] [Uncovering 2-D toroidal representations in
grid cell ensemble activity during 1-D
behavior](https://www.biorxiv.org/content/10.1101/2022.11.25.517966v1.full.pdf)

- [ ] Get to know
  - [ ] Paper from Catalin
    - [ ] Overview
    - [ ] Detailed
  - [ ] Paper from Andres
    - [ ] Powerpoint Summarys
    - [ ] Detailed
    - [ ] PR from 09.05.23 was send in slack 
  
##  Own Questsions
  - [ ] Connectomex --> Neurons + connections (do they have usefull data forstatistics)

# **Todos**
## <ins>**Intrisic Imaging Pipeline**</ins>
- [ ] **Code Algo Autodetection** for Errors in Data + Solutions
    - [ ] create yaml file for session data

### **Data**
- [x] Connect to Drives
  - [x] [Biozentrum](smb://unibasel.ads.unibas.ch/bz/)
  - [x] [Scicore Biozentrum](smb://toucan-all.scicore.unibas.ch/donafl00-calcium$/) for <ins>Rodrigos data</ins>
  - [x] [Biopz Jumbo](smb://biopz-jumbo.storage.p.unibas.ch/RG-FD02$/_Members/mauser00/Desktop)
    - Middle connection, because Biozentrum is not safe?
    - Data from Steffen &rarr; Catalin/Me process &rarr; Rodrigo for Network Analysis (eg. Density)

- [ ] Important Data: Replicate, Detect, Clean
  - [x] Focus on Clean Datasets from pubs from 2021, than adults and 2022
    - [x] 2021
      - [x] Pups 00608X, 4, 5, 7
      - [x] Adults ,002865 ,003165 ,003343
    - [x] 2022
      - [x] Pubs ,9191, 9192, 10473, 10477
      - [x] Adults ,8497 ,8498 ,8499
    - [ ] 2023
      - [ ] DON-014837 DON-014838 DON-014840 DON-014847 DON-014849 DON-015078 DON-015079

### **Code base**
- Own
  - Animal Class
  - Session Class
  - Vizualizer Class
  - Analyzer Class
- Others code
  - [Notebook for visualize some of the problems]
  - [Mesc to Tiff] 

## <ins>**Manifolds for individual comparisson**</ins>
comment: 
- didn't work immiedately out-of-the-box on Steffen's treadmill data
  - so that's mostly why I was digging into the algorithm and trying to figure out the limitations..

- usefull Variables :
  - [ ] Behaviour
    - [ ] Pupils
    - [ ] Movement
  - [ ] Time

- [ ] Projects (Datasets):
  - [ ] Active Avoidance paradigm (Nathalie)
  - [ ] Sensory Treadmil (Steffen)
    - Treadmil+VR &rarr; pillar counting
    - 3 Different belts 
      - A
      - A'
      - B(lank)

- Methods
  - [ ] Look at Manifolds
    - [ ] [Cebra](https://cebra.ai/)
    - [ ] [MIND](https://www.biorxiv.org/content/10.1101/418939v2.full)
      - Tank Lab (Prinston)
      - [MIND algorithm](https://github.com/catubc/gzenke_mind)
        - Based on IsoMap + Metrics
        - Evidence Akkumulation + Manifold extraction
    - [ ] [pi-VAE](https://arxiv.org/abs/2011.04798)
  - [ ] Decoder
    - [x] Baeysian-Decoder (Done by Catalin)
    - [x] PCA+KNN (Done by Catalin)


## <ins>**Package CaBinCorr Paper (until end of summer)**</ins>
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

- [ ] Create Class 
  - [x] Visualizer
    - [x] Traces
    - [x] Raster
    - [x] Histograms
    - [x] KDE
  - [x] Analyzer
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


## <ins>**BMI**</ins>
- [ ] BMI GUI
  - [ ] Matplotlib slow &rarr; PyQT? PyQT-Graph? what is faster
- [ ] Analysis needs to be standardized
- [ ] If I have enough time:
  - [ ] motions correction by
    - [ ] Separate Contours for BMI into multiple blocks 
    - [ ]  motion correct individual block (because brain moves differently, because different pressure on Brain)

## <ins>**Active Avoidance paradigm**</ins>
## <ins>**Nathalie Mouse in a Box**</ins>
## <ins>**Nathalie Volition/Imagination**</ins>




## Sources
[Mesc to Tiff]: https://github.com/donatolab/manifolds/tree/maindonlabtools/renan_tiff_process
[Suit2p]: https://github.com/MouseLand/suite2p
[20220319]:smb://unibasel.ads.unibas.ch/bz/RG%20Donato/Microscopy/Steffen/Experiments/DON-009192/20220319
[Notebook for visualize some of the problems]: https://github.com/donatolab/manifolds/blob/main/donlabtools/intrinsic_dynamics_project/Visualize_suite2p_concatenated_data.ipynb

![pic](../Catalin/Suit2P_test_04.05.2023.jpg)
<img src="../Catalin/Suit2P_test_04.05.2023.jpg" alt="Getting started" />
<img src="https://www.mylifeorganized.net/i/products/notebook.png" style="width: 180px">