# Medical-Image-Segmentation-using-One-to-Many-U-Net-model
In this paper, we propose and evaluate a dilated One-to-Many U-Net deep learning model that addresses these challenges. 

Medical image processing applications typically demand highly accurate image segmentation.
However, existing segmentation approaches exhibit performance degradation when faced with diverse
medical imaging modalities and varied segmentation target sizes. In this paper, we propose and evaluate
a dilated One-to-Many U-Net deep learning model that addresses these challenges. The proposed model
comprises of four rows of encoder-decoder modules, with each module consisting of three trainable blocks
with different layers. The last three rows of the U-Net are extended versions of the three blocks in the first
row, with the encoder-decoder blocks connected through the skip connections to the previous rows. The
outputs of the last blocks from the last three rows in the decoder are concatenated, and finally, a dilation
network is employed to improve the small target segmentation in different medical images. Two datasets
have been used for the evaluation: the HC18 grand challenge ultrasound dataset for fetal head segmentation
and the Multi-site MRI dataset, including the BIDMC and HK sites, for prostate segmentation in MRI
images. The proposed approach achieved Dice and Jaccard coefficients of 96.54% and 93.93%, respectively,
for the HC18 grand challenge dataset, 96.76% and 93.97% for the BIDMC site dataset, and 92.58% and
86.96% for the HK site dataset. Statistical analyses showed that the proposed model outperformed several
other U-Net-based models.

Below figure llustrates the proposed dilated One-to-Many model.
<img width="1091" alt="Screenshot 2024-12-22 at 9 51 09 AM" src="https://github.com/user-attachments/assets/5e894959-4f67-488c-8ae8-88206ada8b44" />

# Datasets: 
We conducted the experiments on two datasets with different
image types, including the HC18 Grand challenge dataset and the Multi-site MRI dataset. 
HC18-Grand challenge dataset: This public dataset
comprises 1334 two-dimensional ultrasound images to measure
the fetal HC parameter. In this dataset, there are 999
images manually annotated by an expert and 335 unannotated
images. The resolution of all ultrasound images is 800 by 540
pixels, with a pixel size ranging from 0.052 to 0.326 mm.
Multi-site MRI dataset: The T2-weighted MRI dataset
was collected for prostate identification purposes. This
dataset is comprised of two public sources - sites E and F. 
<img width="546" alt="Screenshot 2024-12-22 at 9 56 30 AM" src="https://github.com/user-attachments/assets/844c65f0-488a-420a-a42c-68cac033ce02" />

# Results:
<img width="634" alt="Screenshot 2024-12-22 at 9 57 57 AM" src="https://github.com/user-attachments/assets/22e93095-722d-4957-9c4e-d91451c0934b" />

<img width="628" alt="Screenshot 2024-12-22 at 9 58 12 AM" src="https://github.com/user-attachments/assets/295b88ac-31d9-488c-80f4-5c5f803b309d" />



