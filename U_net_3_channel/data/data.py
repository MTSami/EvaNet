import os
import re
import numpy as np

import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms

import math
import sys

sys.path.append('../')

import config

################################################################################
## Create Torch Dataset from cropped images
################################################################################

class ElevationDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, transforms):
        
        self.data_path = data_path
        self.transforms = transforms
        
        self.feature_files = os.listdir(data_path)
        self.feature_files = [file for file in self.feature_files if file.endswith(".npy") and re.match(".*features.*", file) ]
        
        self.data_len = len(self.feature_files)
        
        assert self.data_len>0, "No data found!!"
    
    
    def normalize(self, data):        
        # global_max = 76.05
        # global_min = -4.965000152587891
        
        global_max = config.GLOBAL_MAX
        global_min = config.GLOBAL_MIN
        
        normalized_data = (data-(global_min))/(global_max-global_min)
        
        assert np.min(normalized_data) >=0, "Normalized value should be greater than equal 0"
        assert np.max(normalized_data) <=1, "Normalized value should be lesser than equal 1"
        
        return normalized_data
    
    
    
    def __getitem__(self, idx):
        
        self.data_dict = dict()
        
        ## Get the feature patch
        self.feature_file = self.feature_files[idx]
        self.feature_data = np.load(os.path.join(self.data_path, self.feature_file))
        
        ## Get the corresponding label
        self.label_file = re.sub("features", "label", self.feature_file)
        ##print("file_name: ", self.label_file)
        self.label_data = np.load(os.path.join(self.data_path, self.label_file)).astype('int')
        
        
        ## Seperate elevation data from RGB
        self.disaster_rgb = self.feature_data[:,:, :3].astype('uint8')
        self.elev_data = self.feature_data[:,:, 3].astype('float32')
        self.elev_data = np.expand_dims(self.elev_data, 0).astype('float32')
        # print("self.elev_data: ", self.elev_data.shape)
        self.regular_rgb = self.feature_data[:,:, 4:].astype('uint8')
        
        
        ## Format labels for Loss function
        """
            Elev_loss function label format: Flood = 1, Unknown = 0, Dry = -1 
            Data label format: Flood = 1, Dry = 0, Unknown = -1
            Label shape: B, C, H, W
        """      
        # self.flood = np.where(self.label_data == 1, 1, 0).astype('int')
        # self.dry = np.where(self.label_data == 0, -1, 0).astype('int')
        # self.formatted_label_data = self.flood + self.dry
        # self.formatted_label_data = np.expand_dims(self.formatted_label_data, 0).astype('int')
        
        
        ### For testing labels with CE loss 0: Unknown, 1: Flood, 2: Dry
        self.formatted_label_data = np.where(self.label_data == 0, 2, self.label_data).astype('int')
        self.formatted_label_data = np.where(self.label_data == -1, 0, self.formatted_label_data).astype('int')
        
        
        
        ## Merge Disaster and regular time RGB
        #self.rgb_data = np.concatenate((self.disaster_rgb, self.regular_rgb), axis = -1)
        self.rgb_data = self.disaster_rgb
        ## print("self.rgb_data.shape: ", self.rgb_data.shape)
        
        
        ## Apply torchvision tranforms to rgb data
        self.transformed_rbg = self.transforms(self.rgb_data)
        ## print("self.transformed_rbg: ", self.transformed_rbg.shape)
        ## print(torch.max(self.transformed_rbg))
        ## print(torch.min(self.transformed_rbg))
        
        
        ## Normalize to elev_data
        self.norm_elev_data = self.normalize(self.elev_data)
        # print("self.elev_data: ", self.elev_data.shape)
        # print(np.max(self.elev_data))
        # print(np.min(self.elev_data))
                
        
        ## Put all data in one dictionary
        self.data_dict['filename'] = self.feature_file
        self.data_dict['rgb_data'] = self.transformed_rbg
        self.data_dict['elev_data'] = self.elev_data
        self.data_dict['norm_elev_data'] = self.norm_elev_data
        self.data_dict['labels'] = self.formatted_label_data
        
        return self.data_dict
    
    
    def __len__(self):
        return self.data_len
        

def get_dataset(cropped_data_path):
    
    training_transforms = []
    training_transforms += [transforms.ToTensor()]
    training_transforms += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))]
    
    
    data_transforms = transforms.Compose(training_transforms)
    
    elev_dataset = ElevationDataset(cropped_data_path, data_transforms)
    
    return elev_dataset
################################################################################
        