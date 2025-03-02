# Import Packages

import numpy as np
import scipy as sp
import pandas as pd

import torch
import torch.nn            as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchinfo
import torchvision.transforms.v2 as v2
import skimage.io as ski

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit

import os
from typing import Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

from DLfunctions import Train_model

import PIL

# Upload CSV
csv_file_path = r'/home/diana/train.csv'
df = pd.read_csv(csv_file_path)
# path of data folder
data_folder_path =  r'/home/diana/train-jpegs'

#Filter patients with between 200 to 250 images
image_counts = df.groupby('StudyInstanceUID')['SOPInstanceUID'].count()
filtered_patients = image_counts[(image_counts >= 30) & (image_counts <= 1000)]
filtered_patient_ids = filtered_patients.index
filtered_df = df[df['StudyInstanceUID'].isin(filtered_patient_ids)]
patient_status_counts = filtered_df.drop_duplicates(subset='StudyInstanceUID')['negative_exam_for_pe'].value_counts()

print(f"Number of patients with 200 to 250 images: {len(filtered_patient_ids)}")
print(f"Number of healthy patients: {patient_status_counts.get(1, 0)}")
print(f"Number of sick patients: {patient_status_counts.get(0, 0)}")

#Balance Data

## all data - taking half of the neg and all pos (using JPGs)
# delete duplicates - StudyInstanceUID
unique_patients_df = filtered_df.drop_duplicates(subset='StudyInstanceUID')
patient_counts = unique_patients_df['negative_exam_for_pe'].value_counts()
balanced_unique = filtered_df.drop_duplicates(subset='StudyInstanceUID')

#num of neg to take as num of positives
neg_bal_patients = balanced_unique[balanced_unique['negative_exam_for_pe'] == 1].head(patient_counts[0])
#num of pos - all pos
pos_bal_patients = balanced_unique[balanced_unique['negative_exam_for_pe'] == 0]

neg_bal_patients_ids = neg_bal_patients['StudyInstanceUID'].tolist()
pos_bal_patients_ids = pos_bal_patients['StudyInstanceUID'].tolist()

print(f"num of patients without PE: {len(neg_bal_patients)}")
print(f"num of patients with pe: {len(pos_bal_patients)}")

# create df
balanced_patients=pd.concat([neg_bal_patients, pos_bal_patients], ignore_index=True)
#list of all IDs
ids_balanced=neg_bal_patients_ids + pos_bal_patients_ids
# extracting from original csv to a new working file
df_balanced=filtered_df[filtered_df['StudyInstanceUID'].isin(ids_balanced)]


# Find lunges images
# Take 20 images with highest probability to be positie in selection model

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
# Load the pre-trained model (ensure the model architecture matches)
checkpoint = torch.load('Best_Model_F1_6_select.pth', weights_only=False)
model = GetModel(num_classes=1)  # Initialize the model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

def select_top_X_images_per_patient(df_balanced, root_dir):
    # List to store the selected rows
    selected_rows = []

    # Group by StudyInstanceUID (Patient ID)
    grouped = df_balanced.groupby('StudyInstanceUID')

    # Loop over each patient group
    for patient_id, group in grouped:
        group_series = group['SeriesInstanceUID'].unique()[0]  # Get the unique series for the patient

        # Path to the patient's image directory
        path = os.path.join(root_dir, patient_id, group_series)

        # Get the image filenames from the directory
        slices = os.listdir(path)

        # Filter and sort the filenames by the position part (e.g., 001_SERIAL_NUMBER)
        valid_slices = []
        for s in slices:
            try:
                # Extract the numeric part before the first underscore
                position = int(s.split('_')[0])  # Extracts the "001" from "001_SERIAL_NUMBER"
                valid_slices.append((position, s))
            except ValueError:
                # Skip files that don't follow the expected pattern
                continue

        # Sort the valid slices by the numeric position
        valid_slices.sort(key=lambda x: x[0])

        # Extract just the filenames (sorted by position)
        sorted_slices = [s[1] for s in valid_slices]

        # Initialize a list to store image predictions
        image_predictions = []

        # Process each image
        for img_name in sorted_slices:
            # Load the image
            img_path = os.path.join(path, img_name)
            image =  PIL.Image.open(img_path).convert('RGB')
            
            # Preprocess the image to match the model's input requirements
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

            # Get the model's prediction
            with torch.no_grad():
                prediction = model(input_tensor)

            # Since it's a binary classification, apply sigmoid to get the probability
            probability = torch.sigmoid(prediction).item()

            # Store the prediction with the corresponding image filename
            image_predictions.append((probability, img_name))

        # Sort the images by predicted probability in descending order
        image_predictions.sort(key=lambda x: x[0], reverse=True)

        # Select the top X images with the highest probability
        selected_slices = [s[1] for s in image_predictions[:60]]

        # Print selected slices for debugging
        print(f"Selected slices: {selected_slices}")

        # Strip the serial number and .jpg extension from selected_slices to get the SOPInstanceUID
        db_selected_slices = [s.split('_')[1].replace('.jpg', '') for s in selected_slices]

        # Print the cleaned SOPInstanceUIDs for debugging
        print(f"DB selected SOPInstanceUIDs: {db_selected_slices}")

        # Filter the group to keep only the rows where SOPInstanceUID matches the cleaned filenames
        selected_group = group[group['SOPInstanceUID'].isin(db_selected_slices)]

        # Append these rows to the list
        selected_rows.append(selected_group)

    # Concatenate all the selected rows into a new DataFrame
    new_df = pd.concat(selected_rows).reset_index(drop=True)

    return new_df

# Create a new DataFrame with only the top 20 images per patient
df_balanced = select_top_X_images_per_patient(df_balanced, root_dir=data_folder_path)

# Set the Device
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The chosen device: {TORCH_DEVICE}')

# Flipping the values in the 'negative_exam_for_pe' column so that 1 will be positive and 0 negative
df_balanced['negative_exam_for_pe'] = df_balanced['negative_exam_for_pe'].apply(lambda x: 1 if x == 0 else 0)

print(df_balanced['negative_exam_for_pe'].value_counts())




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
        
        
        # Check if the image has only one channel, convert it to 3 channels
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = np.stack([image] * 3, axis=-1)

        # Normalize    
        imgSize = 224
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

        # Split into three separate 1-channel images
        image1 = image[0:1, :, :]  # Extract the first channel
        image2 = image[1:2, :, :]  # Extract the second channel
        image3 = image[2:3, :, :]  # Extract the third channel        
        
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        image3 = image3.to(self.device)           

        label = self.annotations['negative_exam_for_pe'].iloc[idx]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return (image1, image2, image3), label







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
            
        # Check if the image has only one channel, convert it to 3 channels
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = np.stack([image] * 3, axis=-1)

        # Normalize    
        imgSize = 224
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

        # Split into three separate 1-channel images
        image1 = image[0:1, :, :]  # Extract the first channel
        image2 = image[1:2, :, :]  # Extract the second channel
        image3 = image[2:3, :, :]  # Extract the third channel  

        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        image3 = image3.to(self.device)
                                  
        label = self.annotations['negative_exam_for_pe'].iloc[idx]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0).to(self.device)
     
        return (image1, image2, image3), label



### Split train test 
# Split train test according to patient
group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get the indices for train and test sets based on groups
train_index, val_index = next(group_splitter.split(df_balanced, df_balanced['negative_exam_for_pe'], groups=df_balanced['StudyInstanceUID']))

#create train and test df's based on the indices
train_df, val_df = df_balanced.iloc[train_index,:], df_balanced.iloc[val_index,:]  


# Creat Datasets
train_dataset = CustomTrainDataset(annotations=train_df, root_dir=data_folder_path)
val_dataset = CustomValDataset(annotations=val_df, root_dir=data_folder_path)
# Check Data
print(f'Training data size: {len(train_dataset)}')
print(f'Testing data size: {len(val_dataset)}')



# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last= True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)


#1st batch of train dataset
train_images, train_labels = next(iter(train_loader)) #<! PyTorch Tensors

print(f'The number of tensors in the batch: {len(train_images)}')
print(f'Shape of the first channel tensor: {train_images[0].shape}')
print(f'Shape of the second channel tensor: {train_images[1].shape}')
print(f'Shape of the third channel tensor: {train_images[2].shape}')
print(f'The batch labels dimensions: {train_labels.shape}')
print(f'The batch labels unique values: {train_labels.unique()}')


#1st batch of test dataset
test_images, test_labels = next(iter(test_loader)) #<! PyTorch Tensors

print(f'The number of tensors in the batch: {len(train_images)}')
print(f'Shape of the first channel tensor: {test_images[0].shape}')
print(f'Shape of the second channel tensor: {test_images[1].shape}')
print(f'Shape of the third channel tensor: {test_images[2].shape}')
print(f'The batch labels dimensions: {test_labels.shape}')
print(f'The batch labels unique values: {test_labels.unique()}')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Initialize downsample if needed
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust shortcut connection if downsampling is required

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out += identity  # Add the residual (shortcut) connection
        out = self.relu(out)

        return out

class CustomConvNetWithResiduals(nn.Module):
    def __init__(self, in_channels, num_classes=1, dropout_rate=0.2):
        super(CustomConvNetWithResiduals, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create downsample layers for residual blocks
        self.residual_block1_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )
        
        self.residual_block2_downsample = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024)
        )

        # Residual blocks
        self.residual_block1 = ResidualBlock(256, 512, stride=2, downsample=self.residual_block1_downsample, dropout_rate=dropout_rate)
        self.residual_block2 = ResidualBlock(512, 1024, stride=2, downsample=self.residual_block2_downsample, dropout_rate=dropout_rate)

        # Global average pooling and fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Pass through initial convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.maxpool(x)

        # Pass through the residual blocks
        x = self.residual_block1(x)
        x = self.residual_block2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x



class SiameseNetwork(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.2):
        super(SiameseNetwork, self).__init__()
        
        # Three identical networks for the three channels
        self.branch1 = CustomConvNetWithResiduals(in_channels=1, num_classes=512, dropout_rate=dropout_rate)
        self.branch2 = CustomConvNetWithResiduals(in_channels=1, num_classes=512, dropout_rate=dropout_rate)
        self.branch3 = CustomConvNetWithResiduals(in_channels=1, num_classes=512, dropout_rate=dropout_rate)
        
        # Fully connected layers after concatenation
        self.fc1 = nn.Linear(512 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x1, x2, x3):
        
        # Process each branch
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)
        
        # Concatenate the outputs from the three branches
        combined = torch.cat((out1, out2, out3), dim=1)
        
        # Pass through the final fully connected layers
        combined = self.fc1(combined)
        combined = nn.ReLU()(combined)
        combined = self.fc2(combined)
        
        return combined

def GetModel(num_classes=1, dropout_rate=0.2) -> nn.Module:
    return SiameseNetwork(num_classes=num_classes, dropout_rate=dropout_rate)



oModel = GetModel()

oModel = oModel.to(TORCH_DEVICE) #<! Transfer model to device
train_images = (train_images[0], train_images[1], train_images[2])
# Pass the tuple of tensors as input to `torchinfo.summary`
torchinfo.summary(oModel, input_data=train_images, device=TORCH_DEVICE)
    

#Loss function 
# Fix unbalance images
# Compute with function: compute_class_weight
train_labels_cpu = train_labels.cpu().numpy() #need to pass to numpy from torch for compute class function 
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_cpu), y=train_labels_cpu.flatten())
class_weights = torch.tensor(class_weights, dtype=torch.float)

hL=nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

hL=hL.to(TORCH_DEVICE)


TENSOR_BOARD_BASE   = 'TB'

nEpochs=20

oModel = oModel.to(TORCH_DEVICE)

# Initial optimizer and scheduler
learnRate = 4e-4
oOpt = torch.optim.AdamW(oModel.parameters(), lr=6e-4, betas=(0.9, 0.99), weight_decay=1e-4) 
oScd = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr=learnRate, total_steps=nEpochs * len(train_loader))

# Train the model
oRunModel, history = Train_model(model=oModel, train_loader=train_loader, val_loader=test_loader, criterion=hL, optimizer=oOpt, scheduler=oScd, num_epochs=nEpochs, device=TORCH_DEVICE, is_binary=True, save_metric='f1',l1_lambda=None, oTBLogger=None)


    











        
       

        






