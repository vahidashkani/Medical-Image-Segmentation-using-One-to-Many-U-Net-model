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

<img width="1091" alt="Screenshot 2024-12-22 at 9 51 09 AM" src="https://github.com/user-attachments/assets/9f18c8e8-6ea1-423f-8bce-cf0d4d34ce58" />
