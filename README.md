# Brain Tissue Segmentation from Magnetic Resonance Images Using Deep Learning

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
