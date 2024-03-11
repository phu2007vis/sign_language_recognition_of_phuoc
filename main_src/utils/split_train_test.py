
import os
import numpy as np
import shutil
import math

input_folder = r"/work/21013187/phuoc_sign/dataset/data_sign_croped_2_raw"
output_folder = "/work/21013187/phuoc_sign/dataset/data_sign_croped_2_raw_train_test_split"
train_folder  = os.path.join(output_folder,"train")
test_folder  = os.path.join(output_folder,"test")
train_thresh = 0.8
total = 0
for action_name in  os.listdir(input_folder):
    sub_int_folder = os.path.join(input_folder,action_name)
    list_file_names =  os.listdir(sub_int_folder)
    num_train_sample = math.ceil(train_thresh*len(list_file_names))
    count = 0
    sub_test_folder = os.path.join(test_folder,action_name)
    sub_train_folder =  os.path.join(train_folder,action_name)
    os.makedirs(sub_test_folder,exist_ok=True)
    os.makedirs(sub_train_folder,exist_ok=True) 
    for file_name in list_file_names:
        in_path = os.path.join(sub_int_folder,file_name)
        if count <= num_train_sample:
            
            file_path = os.path.join(sub_train_folder,file_name)
        else:
            file_path = os.path.join(sub_test_folder,file_name)
        count+=1
        total+=1
        shutil.copy(in_path,file_path)
    

        

            
            

       
    
