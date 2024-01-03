# Phước-siuuuu Project

## Overview
Briefly describe what your project is about and its purpose.

## Installation
To set up your project, follow these steps:

1. **Install Python Packages:**
   ```bash
   pip install -r requirements.txt
2. **data preparation (for trainning)**
   - First extract the frames in video  
   - Format data with this structure: 
   ```
   -----root_folder: 
         -----action1: 
            ----video1.mp4 
            ----video2.mp4
            ....
         -----action1:
            ----video1.mp4
            ----video2.mp4
            ....
         ...
   ```
   - Note that: 
      - Root_folder,video.mp4 : can be any name
      - Action : is class name
   - In  main_src/data_preparation/pre_process.py file
      - Change  data_root path with root_folder_path
      - Change output folder 
   - And run:

   ```bash
   python main_src/data_preparation/pre_proces.py
   ```
   - if you want to test dataloader 
   - change folder_path in dataloader.py in datapreparation
   - And run:
   ```bash
   python main_src/data_preparation/Dataloader.py
    ```