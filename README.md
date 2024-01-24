# Brain Tissue Segmentation from Magnetic Resonance Images Using Deep Learning

Author(s): **Mohammad Imran Hossain**, Muhammad Zain Amin
<br>University of Girona (Spain), Erasmus Mundus Joint Master in Medical Imaging and Applications

**Abstract:** The segmentation of brain tissues in Magnetic Resonance Imaging (MRI) is vital for investigating neurodegenerative diseases such as Alzheimer's, Parkinson's, and Multiple Sclerosis (MS). This process goes beyond accurate diagnosis, enabling quantitative analysis and fostering advancements in personalized medicine and targeted therapies. This project explores the effectiveness of several deep learning models—U-Net, nnU-Net, and LinkNet—in segmenting MRI brain tissues using the IBSR18 dataset. Utilizing pre-trained weights from ImageNet, including ResNet 34 and ResNet 50 as backbones for U-Net and LinkNet, enhances the models' performance and feature extraction capabilities. Performance assessment reveals the superior capability of 3D nnU-Net, achieving an average mean **Dice Score** of **0.937 ± 0.012**. Noteworthy, 2D nnU-Net excels in **Hausdorff Distance (5.005 ± 0.343)** and **Absolute Volumetric Difference (3.695 ± 2.931)**. This comprehensive analysis underscores the unique strengths of each model in different facets of brain tissue segmentation.

# Dataset 
The dataset, **IBSR 18**, used for this brain tissue segmentation challenge is a publicly available dataset by the Center for Morphometric Analysis at Massachusetts General Hospital in the United States of America. The dataset consists of **18 T1-weighted** volumes with different slice thicknesses and spacing.

- Training Set  : 10
- Validation Set: 05
- Test Set      : 03

# Methodology
The brain tissue segmentation from MRI using Deep Learning consists of several key steps including pre-processing, augmentation, patch extraction, training models, and prediction. Each of the mentioned sections is explained with necessary figures and relevant information.

### Pre-processing
Segmenting brain tissue from medical images faces challenges due to scanner variations, causing uneven intensity, contrast, and noise. Essential preprocessing steps, including bias field correction and anisotropic diffusion, are pivotal for achieving uniform and precise tissue segmentation. While IBSR 18 volumes are already skull-stripped, we applied **N4 Bias Field Correction** and **Anisotropic Diffusion** to denoise intensities. Additionally, we performed **Image Normalization** aligns intensity scales, facilitating consistent processing for deep learning methods.

<p align="center">
  <img src="https://github.com/imran-maia/IBSR_18_BraTSeg_Deep_Learning/assets/122020364/b1639ea9-dfdf-45d3-bf7f-cd7294e74104" width="700" alt="Pre-processed Image">
</p>
<p align="center">Figure 1: Pre-processed Image.</p>

### Patch Extraction
Training deep learning models for segmentation with higher-dimensional images presents computational challenges. To address this, we employed patch extraction from 3D volumes in the IBSR 18 dataset, specifically for U-Net and LinkNet architectures. Using 32×32 patches from the full 3D volumes (256×128×256) and their corresponding ground truths during training, the patch extraction process involved the following steps:

<p align="center">
  <img src="https://github.com/imran-maia/IBSR_18_BraTSeg_Deep_Learning/assets/122020364/79ff8b19-7281-4660-839e-c446c8dd2d3e" width="400" alt="Extracted Patches">
</p>
<p align="center">Figure 2: Extracted Patches.</p>


- Define patch size (32×32) and stride size (32×32).
- Create a brain tissue mask to isolate tissue regions from the background in the volumes.
- Extract patches from the specified brain tissue regions for both training images and their corresponding labels.
  
This approach streamlined the training process for U-Net and LinkNet, optimizing computational resources without compromising segmentation quality.

### Data Augmentation

Data augmentation is a crucial step in our deep learning workflow, particularly in optimizing U-Net and LinkNet models for brain tissue segmentation. We applied robust augmentation techniques to artificially expand the training data. These techniques are: 
- Rotation.
- Shift.
- Scaling
- Shear Transformation.
- Horizontal and Vertical Flipping.

This diverse augmentation strategy enhances the robustness of models to different orientations and anomalies, contributing significantly to the accuracy and reliability of brain tissue segmentation.

### Model Training and Prediction

Deep learning excels in medical image analysis, particularly in brain tissue segmentation. Utilizing complex neural network architectures such as CNN, U-Net, nnU-Net, LinkNet, and SegFormer deep learning models demonstrate superior performance, handling intricate patterns and variability in medical images. This shift towards AI-driven methods signifies a significant advancement in neuroimaging and neurological disease diagnosis, showcasing the increasing reliance on innovative machine-learning approaches in healthcare. In this project, we have leveraged three different deep learning architectures (U-Net, nnU-Net, and LinkNet) for brain tissue segmentation. A short description of training and prediction of each mentioned model is explained below.


<p align="center">
  <img src="https://github.com/imran-maia/IBSR_18_BraTSeg_Deep_Learning/assets/122020364/6bda2b57-e093-472c-af78-93e2a626ccc4" width="900" alt="Deep Learning Pipeline">
</p>
<p align="center">Figure 3: Pipeline for Brain Tissue Segmentation Using Deep Learning.</p>

**U-Net:** We have investigated various U-Net architectures, including customized models and models with pre-trained backbones, for brain tissue segmentation in magnetic resonance images. Comparing our customized models, the first is straightforward with single 2D Convolution layers, the second adds complexity with extra Convolution layers and spatial information-preserving connections, and the third further enhances integration with additional concatenation steps. In addition to customized U-Net models, we explored architectures with pre-trained backbones such as ResNet 34 and ResNet 50 from the segmentation models Python library. These backbones, based on the ImageNet dataset, serve as the encoding path, extracting features. The decoder path, tailored for U-Net, complements up-sampling layers and convolutional processes. Integrating pre-trained backbones enhances the complexity of U-Net, often improving biomedical imaging segmentation, such as brain tissue segmentation in MR images.

**LinkNet:** In addition to the U-Net deep learning architecture, we also explored LinkNet, a high-speed deep learning model, widely used for real-time semantic segmentation due to its first computational capacity. We used the pre-trained ResNet 34 model as an encoder backbone for extracting features efficiently during training. This strategy aligns with the streamlined approach we exectued with the U-Net architecture using the Segmentation Model Python library.

**nn-UNet:** nnU-Net is a deep learning model widely used for semantic segmentation, especially in the biomedical imaging field. The model has been designed to autonomously adapt and optimize itself by encompassing the customization of preprocessing methods, network architecture, training procedures, and post-processing techniques for any assigned new task. Therefore, users do not need to be concerned about the required parameters to achieve desired outcomes during the training phase. By default, the nn-UNet provides three different configurations such as:

- Two-dimensional (2D) nn-UNet.
- Three-dimensional (3D) nn-UNet for operating the entire resolution of images.
- Three-dimensional (3D) Cascade nn-UNet where the first U-Net works with down-sampled images and the second is trained to enhance the full-resolution segmentation maps produced by the former.

However, in this project, we have explored the first two configurations 2D nn-UNet and 3D nn-UNet. Users are required to follow some necessary steps to create the working environment for preparing datasets for the nnU-Net training and prediction. The steps begin with creating the directories and maintaining specific formats for the nnU-Net model. After that, we prepared the dataset (Jason format) by specifying the imaging modality (MRI) and labels (Background, CSF, GM, WM) for the training. Furthermore, we executed the pre-processing command to convert the dataset into the nnU-Net format. Moreover, we performed training indicating the desired configuration (2D nnUNet) with the command. Finally, after the successful training of the model, the unseen test dataset was predicted.


# Results
To assess the performance of our brain tissue segmentation models, we employed three key metrics: Dice Coefficient Score (DSC), Hausdorff Distance (HD), and Absolute Volumetric Distance (AVD). The visual representation of predicted segmentation results from each deep-learning model is illustrated in Figure 3. 

<p align="center">
  <img src="https://github.com/imran-maia/IBSR_18_BraTSeg_Deep_Learning/assets/122020364/686014f4-7b50-44a0-a29d-a2cc4221e231" width="500" alt="Segmentation Result">
</p>
<p align="center"><b>Figure 3:</b> Predicted Segmentation Results of Deep Learning Models.</p>
<br>
<br>

Beyond visualizing the segmented results, a comprehensive performance analysis is presented in Table 1, outlining the computed metrics for each model. After analyzing all the computed metrics, it can be said that the nn-UNet (2D and 3D) outperformed other deep learning models U-Net and LinkNet and provided outstanding segmentation results.

<br>
<p align="center"><b>Table 1:</b> Performance Analysis of Deep Learning-Based Brain Tissue Segmentation.</p>
<p align="center">
  <img src="https://github.com/imran-maia/IBSR_18_BraTSeg_Deep_Learning/assets/122020364/ba4958da-0121-4598-8dcf-67d5f6dd4892" width="700" alt="Segmentation Result">
</p>





