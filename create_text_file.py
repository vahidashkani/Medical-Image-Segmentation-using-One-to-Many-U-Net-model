import glob
import random



image_files = glob.glob("/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/UCL_augmented/image_png/*")
mask_files = glob.glob("/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/UCL_augmented/mask_png/*")
image_files.sort()
mask_files.sort()

temp = list(zip(image_files, mask_files)) 
random.shuffle(temp) 
image_files, mask_files = zip(*temp) 


train_idx = len(image_files)*0.80

for i in range(len(image_files)):
    if i < train_idx:
        with open("/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/UCL_augmented/train_augmented_UCL.txt", 'a') as f:
            f.write(image_files[i] + " " + mask_files[i] + "\n")
    else:
        with open("/content/drive/MyDrive/Codes_HC18_Grand_Challenge_dataset/UCL_augmented/val_augmented_UCL.txt", 'a') as f:
            f.write(image_files[i] + " " + mask_files[i] + "\n")        
