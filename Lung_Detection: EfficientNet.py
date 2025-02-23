# Import Packages

import numpy as np
import pandas as pd
import torch
import torch.nn            as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchinfo
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import torchvision.models as models
import skimage.io as ski
from sklearn.model_selection import GroupShuffleSplit
import os
from typing import Callable, Dict, Generator, List, Optional, Self, Set, Tuple
import matplotlib.pyplot as plt
import skimage as ski
from DLfunctions import Train_model

# Upload CSV
csv_file_path = r'/home/diana/train.csv'
df = pd.read_csv(csv_file_path)
# path of image folder
data_folder_path =  r'/home/diana/train-jpegs'


# Set the Device
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #<! for first device: `cuda:0` 
print(f'The chosen device: {TORCH_DEVICE}')

#Balance Data

#num of neg to take as num of positives
neg_bal_patients = df[df['pe_present_on_image'] == 0].head(len(df[df['pe_present_on_image'] == 1]))
#num of pos - all pos
pos_bal_patients = df[df['pe_present_on_image'] == 1]
neg_bal_patients_ids = neg_bal_patients['StudyInstanceUID'].tolist()
pos_bal_patients_ids = pos_bal_patients['StudyInstanceUID'].tolist()
print(f"num of images without PE: {len(neg_bal_patients)}")
print(f"num of images with pe: {len(pos_bal_patients)}")

#df
balanced_patients=pd.concat([neg_bal_patients, pos_bal_patients], ignore_index=True)

class CustomTrainDataset(Dataset):
    def __init__(self, annotations, root_dir, device=TORCH_DEVICE): 
        self.annotations = annotations
        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        

        # Group by patient ID
        self.groups = self.annotations.groupby('StudyInstanceUID')
        self.patient_ids =list(self.groups.groups.keys())
        

    def __len__(self):
         return len(self.annotations)

    def load_image(self, patient_id, idx):
        
        #get patient data
        group_study = self.groups.get_group(patient_id) #patient id
        group_series = (group_study['SeriesInstanceUID'].unique())[0] #series id
        
        #path of patient files
        path=os.path.join(self.root_dir, patient_id, group_series)
        
        #file list of patients
        patient_files = os.listdir(path)
        
        patient_file_names = [f.split('_')[-1].split('.')[0].strip() for f in patient_files]
        
        #find the correct file in file list
        i_idx = patient_file_names.index(self.annotations['SOPInstanceUID'].iloc[idx])
        
        #path of image
        path=os.path.join(self.root_dir, patient_id, group_series, patient_files[i_idx])
        
        #reading image
        image = ski.io.imread(path)
        # image = imread(path)
        
        
        # Check if the image has only one channel, convert it to 3 channels
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = np.stack([image] * 3, axis=-1)

        # Normalize    
        imgSize = 224
        # imgSize = 160
        vMean = np.array([0.485, 0.456, 0.406])
        vStd  = np.array([0.229, 0.224, 0.225])

        oPreProcess = v2.Compose([
            v2.ToImage(), 
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(degrees=15),
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            v2.Resize(imgSize),
            v2.CenterCrop(imgSize),
            v2.ToDtype(torch.float32, scale=True),  
            v2.Normalize(mean=vMean, std=vStd),  
        ])

        image = oPreProcess(image)
        image = image.to(self.device)
        
        return image
    
    def __getitem__(self,idx):
        
        idx = int(idx)
        
        # patient num
        patient_id = self.annotations['StudyInstanceUID'].iloc[idx]
        
        # Use the load_slices method for image loading and sampling
        image = self.load_image(patient_id, idx)                    
                

        label = self.annotations['pe_present_on_image'].iloc[idx]

        

        #change to fit the output shape of the model
        label = np.expand_dims(label, axis=0)     
        label = torch.tensor(label, dtype=torch.float32).clone().detach()
        label =label.to(self.device)
        
        return image, label 


class CustomValDataset(Dataset):
    def __init__(self, annotations, root_dir, device=TORCH_DEVICE): 
        self.annotations = annotations
        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        

        # Group by patient ID
        self.groups = self.annotations.groupby('StudyInstanceUID')
        self.patient_ids =list(self.groups.groups.keys())
        

    def __len__(self):
         return len(self.annotations)

    def load_image(self, patient_id, idx):
        
        #get patient data
        group_study = self.groups.get_group(patient_id) #patient id
        group_series = (group_study['SeriesInstanceUID'].unique())[0] #series id
        
        #path of patient files
        path=os.path.join(self.root_dir, patient_id, group_series)
        
        #file list of patients
        patient_files = os.listdir(path)
        
        patient_file_names = [f.split('_')[-1].split('.')[0].strip() for f in patient_files]
        
        #find the correct file in file list
        i_idx = patient_file_names.index(self.annotations['SOPInstanceUID'].iloc[idx])
        
        #path of image
        path=os.path.join(self.root_dir, patient_id, group_series, patient_files[i_idx])
        
        #reading image
        image = ski.io.imread(path)
        # image = imread(path)
        
        
        # Check if the image has only one channel, convert it to 3 channels
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = np.stack([image] * 3, axis=-1)

        # Normalize    
        imgSize = 224
        # imgSize = 160
        vMean = np.array([0.485, 0.456, 0.406])
        vStd  = np.array([0.229, 0.224, 0.225])

        oPreProcess = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean = vMean,std=vStd),
        ])

        image = oPreProcess(image)
        image = image.to(self.device)
        
        return image
    
    def __getitem__(self,idx):        
        idx = int(idx)
        
        # patient num
        patient_id = self.annotations['StudyInstanceUID'].iloc[idx]
        
        # Use the load_slices method for image loading and sampling
        image = self.load_image(patient_id, idx)                    
                
        label = self.annotations['pe_present_on_image'].iloc[idx]
     
        #change to fit the output shape of the model
        label = np.expand_dims(label, axis=0)     
        label = torch.tensor(label, dtype=torch.float32).clone().detach()
        label =label.to(self.device)
        
        return image, label 


### Split train test 

# Split train test according to patient
group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# Get the indices for train and test sets based on groups
train_index, val_index = next(group_splitter.split(balanced_patients, balanced_patients['pe_present_on_image'], groups=balanced_patients['StudyInstanceUID']))
#create train and test df's based on the indices
train_df, val_df = balanced_patients.iloc[train_index,:], balanced_patients.iloc[val_index,:]

# Creat Datasets
train_dataset = CustomTrainDataset(annotations=train_df, root_dir=data_folder_path)
val_dataset = CustomValDataset(annotations=val_df, root_dir=data_folder_path)
# Check Data
print(f'Training data size: {len(train_dataset)}')
print(f'Testing data size: {len(val_dataset)}')

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last= True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)


#1st batch of train dataset
train_images, train_labels = next(iter(train_loader)) #<! PyTorch Tensors

#1st batch of test dataset
test_images, test_labels = next(iter(test_loader)) #<! PyTorch Tensors


### EfficientNet

class ModifiedEfficientNet(nn.Module):
    def __init__(self, num_classes=1):#, dropout_rate=0.6):
        super(ModifiedEfficientNet, self).__init__()
        
        # Load a pre-trained EfficientNet model
        self.efficientnet = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.IMAGENET1K_V1')
        
        # Get the number of input features for the final fully connected layer
        num_features = self.efficientnet.classifier[1].in_features
        
        # Replace the fully connected layer with a new one that has the desired number of output classes
        self.efficientnet.classifier = nn.Sequential(
            # nn.Dropout(p=dropout_rate),  # Add Dropout layer before the fully connected layer
            # nn.BatchNorm1d(num_features),  # Add BatchNorm 
            nn.Linear(num_features, num_classes))

    def forward(self, x, return_feature_maps=False):
        # Forward pass through the EfficientNet
        if return_feature_maps:
            feature_maps = []
            for name, layer in self.efficientnet.features._modules.items():
                x = layer(x)
                feature_maps.append(x)  # Store the feature map after this layer
            return feature_maps
        else:
            x = self.efficientnet(x)  # Final output
            return x
    
    def extract_features(self, x):
        # Extract features before the classifier layer
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

def GetModel(num_classes=1) -> nn.Module:
    return ModifiedEfficientNet(num_classes=num_classes)#, dropout_rate=dropout_rate)

oModel = GetModel()

oModel = oModel.to(TORCH_DEVICE) #<! Transfer model to device
torchinfo.summary(oModel, train_images.shape, device = TORCH_DEVICE)



### Freeze layers
# Step 1: Initially freeze all layers except the last few layers
def freeze_all_but_last_n_layers(model, num_layers_to_unfreeze=2):
    total_layers = len(list(model.named_parameters()))
    for i, (param_name, param) in enumerate(model.named_parameters()):
        if i >= total_layers - num_layers_to_unfreeze or 'fc' in param_name or 'bn' in param_name:
            param.requires_grad = True
        else:
            param.requires_grad = False

# Step 2: Unfreeze additional layers
def gradually_unfreeze_layers(model, num_layers_to_unfreeze):
    layers_unfrozen = 0
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True
            layers_unfrozen += 1
            if layers_unfrozen >= num_layers_to_unfreeze:
                break

# Step 3: Unfreeze all layers and set a very small learning rate for fine-tuning
def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True

#Loss function 
hL=nn.BCEWithLogitsLoss()
hL=hL.to(TORCH_DEVICE)


nEpochs=20
# Initial setup: Freeze all but the last few layers
num_layers_to_unfreeze = 2  # Number of layers to unfreeze each step
freeze_all_but_last_n_layers(oModel, num_layers_to_unfreeze)

# Transfer model to device
oModel = oModel.to(TORCH_DEVICE)

# Initial optimizer and scheduler
learnRate = 4e-3
oOpt = torch.optim.AdamW(oModel.parameters(), lr=6e-4, betas=(0.9, 0.99), weight_decay=1e-3)
oScd = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr=learnRate, total_steps=nEpochs * len(train_loader))

# Training loop with gradual unfreezing
for epoch in range(nEpochs):
    print(f"Epoch {epoch + 1}/{nEpochs}")

    # Train the model
    oRunModel, history = Train_model(model=oModel, train_loader=train_loader, val_loader=test_loader, criterion=hL, optimizer=oOpt, scheduler=oScd, num_epochs=nEpochs, device=TORCH_DEVICE, is_binary=True, save_metric='f1')

    # Gradually unfreeze layers after each epoch or at specific intervals
    if (epoch + 1) % 2 == 0:  # Unfreeze every 2 epochs
        gradually_unfreeze_layers(oModel, num_layers_to_unfreeze=1)

        # Update optimizer to include the newly unfrozen layers
        oOpt = torch.optim.AdamW(filter(lambda p: p.requires_grad, oModel.parameters()), lr=6e-4, betas=(0.9, 0.99), weight_decay=1e-3)
        
        # Recreate the learning rate scheduler
        oScd = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr=learnRate, total_steps=(nEpochs - epoch) * len(train_loader))

# Final unfreeze all layers and fine-tune
unfreeze_all_layers(oModel)
oOpt = torch.optim.AdamW(oModel.parameters(), lr=6e-5, betas=(0.9, 0.99), weight_decay=1e-3)
oScd = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr=6e-5, total_steps=nEpochs * len(train_loader))  # Fine-tune for nEpochs more epochs

# Fine-tuning for the final few epochs
oRunModel, history = Train_model(model=oModel, train_loader=train_loader, val_loader=test_loader, criterion=hL, optimizer=oOpt, scheduler=oScd, num_epochs=nEpochs, device=TORCH_DEVICE, is_binary=True, save_metric='f1')




