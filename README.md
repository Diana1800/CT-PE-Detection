# CT-PE-Detection
Pulmonary embolism is a life-threatening condition that occurs when a blood clot gets lodged in an artery in the lungs. This project aims to detect PE by analyzing CT scan images of patients using deep learning models.

The project consists of two key stages:

  Lung Detection: EfficientNet is used to classify which images show the lungs, regardless of whether the patient is diagnosed as sick. A patient can have over 40% of their images showing pulmonary embolism (PE) but still be considered healthy overall.

  Siamese Network: After selecting from each patiant CT scan the 20 images with the highest likelihood of showing lungs, these images are passed through a Siamese network to predict whether each individual image indicates if the patient is sick or healthy.

Installation

To install the necessary dependencies, run:

bash

pip install -r requirements.txt

The project uses:

    PyTorch
    torchvision
    Mermaid.js (for visualization of the model architecture)
    EfficientNet
    Siamese Networks

Dataset

The CT scan images used in this project are JPEG files pre-processed with different HU normalization values for each channel. Each patient’s data includes multiple CT slices which are fed into the model.
Kaggle Dataset

The dataset used for this project was obtained from the Kaggle repository RSNA STR Pulmonary Embolism Detection (JPEG, 256px) created by Vaillant.

Please note that all the images used in this project are from this pre-processed dataset. Make sure to credit the original dataset when using this data.
Model Architecture

![MODEL BY MERMAID](https://github.com/user-attachments/assets/b9e15740-5747-4551-aa54-4064b2f7b805)

Stage 1: Lung Detection Model

    The first model is based on EfficientNet.
    It classifies images into lung or non-lung.
    Achieved an F1 score of 80% on the validation set.

Stage 2: Siamese Network

    After selecting 20 images with the highest probability of being lung images (both from healthy and sick patients), a Siamese Network is used to compare image pairs.
    Each branch of the Siamese Network is a custom Convolutional Network with residual connections.
    The branches process images separately and then concatenate their features for further analysis.


Training Procedure

    The EfficientNet model is first trained to classify CT images as lung or non-lung.
    After achieving a satisfactory F1 score, the top 20 images per patient are passed to the Siamese Network.
    The Siamese Network is trained to determine whether the image pairs indicate a sick or healthy condition.

Results

    EfficientNet: F1 score of 80% on the validation set for lung detection.
    Siamese Network: TBD after final training and testing.


Credits

The dataset used in this project is sourced from the RSNA STR Pulmonary Embolism Detection dataset. For this project, the original DICOM (DCM) files were modified and converted to JPEG format at 256x256 pixels by Ian Pan. Full credit for the original data goes to the original authors.
https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection
https://www.kaggle.com/datasets/vaillant/rsna-str-pe-detection-jpeg-256

Contributing

If you'd like to contribute to this project, please feel free to submit a pull request.
License

The dataset used in this project is licensed under **CC BY-NC-SA 4.0**. The code for this project is licensed under the MIT License.
