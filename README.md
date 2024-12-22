# Medical-Image-Segmentation-using-One-to-Many-U-Net-model
In this paper, we propose and evaluate a dilated One-to-Many U-Net deep learning model that addresses these challenges. 

In this paper, we propose and evaluate
a dilated One-to-Many U-Net deep learning model that addresses these challenges. The proposed model
comprises of four rows of encoder-decoder modules, with each module consisting of three trainable blocks
with different layers. The last three rows of the U-Net are extended versions of the three blocks in the first
row, with the encoder-decoder blocks connected through the skip connections to the previous rows. The
outputs of the last blocks from the last three rows in the decoder are concatenated, and finally, a dilation
network is employed to improve the small target segmentation in different medical images.
