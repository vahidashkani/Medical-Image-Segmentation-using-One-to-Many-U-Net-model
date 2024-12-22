import numpy as np
import keras
from PIL import Image
import cv2
import nibabel as nib
import pydicom
import random
from skimage import transform
from PIL import ImageOps
import tensorflow

class DataGenerator(tensorflow.keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, data_path, batch_size=1, dim=(256,256), n_channels=1,
                  shuffle=True):
        """Initialization
           1. data_path: the path o train or test txt file (map of images 
              and masks)
           2. batch_size: the training batch_size
           3. dim: dimention of network input
           4. n_channels: the number of input channels (wheter to add 
              iso filter or not)
           5. shuffle: wheter to shuffle the images after each epoch training
        """
        self.dim = dim
        self.batch_size = batch_size
        self.data_path = data_path
        self.n_channels = n_channels
        self.shuffle = shuffle
        
        with open(data_path, 'r') as f:
            self.list_paths = f.readlines()
            
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_paths)))


    def __getitem__(self, index):
        """Generate one batch of data"""
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of x paths
        x_paths = [self.list_paths[k] for k in indexes]
        y_paths = [k.replace("\n","").split(" ")[-1] for k in x_paths]
        x_paths = [k.replace("\n","").split(" ")[0] for k in x_paths]
        
        
        # Perform augmentation if we are in training phase
        if self.data_path=="train.txt":
            augmentation = True
        else:
            augmentation = False
        
        # Turn off the augmentation with frequency=5
        if index % 5 == 0:
            augmentation = False
        
        # Generate data
        X, y = self.__data_generation(x_paths, y_paths, augmentation)

        return X, y


    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def augmentor(self, x, y):
        """Perform multiple augmentation algorithms on input image"""
        
        # Rotation
        ang = random.choice([0,2])
        x = np.rot90(x, ang)
        y = np.rot90(y, ang)
        
        rand_rs = round(np.random.randint(-2, 2)*0.1, 1)
        rand_t = round(np.random.randint(-25, 25), 1)           
        
        # Affine transform
        tf = transform.AffineTransform(rotation=rand_rs, shear=-rand_rs, 
                                       translation=rand_t)
        
        # Warp input and target
        x = transform.warp(x, tf, order=1, preserve_range=True, mode='constant')
        y = transform.warp(y, tf, order=1, preserve_range=True, mode='constant')
        return x, y

    def transform(self, X):
    
        """Normalize input to have mean 1 and std 1"""    
        if np.any(X!=0):
            feature_range=(0,1)
            min_x = X.min()
            max_x = X.max()
            scale_ = (feature_range[1] - feature_range[0]) / (max_x - min_x)
            min_ = feature_range[0] - min_x * scale_
            X *= scale_
            X += min_
        return X


    def nifty_loader(self, path):
        """nii format loader"""
        return nib.load(path).get_data()
 
           
    def pil_loader(self, path):
        """JPG and PNG format loader"""
        img = Image.open(path).convert('L')
        expected_size=[256, 256]
        img.thumbnail((expected_size[0], expected_size[1]))
        # print(img.size)
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        return ImageOps.expand(img, padding)


    def __data_generation(self, x_paths, y_paths, augmentation = False):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Y : (n_samples, *dim, 1)
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, path in enumerate(x_paths):
            
            # Load and preprocess image according to its format
            # it accepts .png, .jpg, .PNG and .JPG suffixes
            if path.endswith(('.png', '.PNG', 'jpg', '.JPG')):
                x = self.pil_loader(path)
                x = np.array(x)
                # Resize to fit input size
                #x = cv2.resize(x, (self.dim[1], self.dim[0]))
                x = x.astype("float32")

                y = self.pil_loader(y_paths[i])
                y = np.array(y)  
                # Resize to fit output size
                #y = cv2.resize(y, (self.dim[1], self.dim[0]))
                
                # Convert y to boolean format
                y = y>(0.5*(np.max(y)-np.min(y))+np.min(y))
                
                # Convert to float32
                y = y.astype("float32")
            
                
            elif path.endswith('.nii.gz'):
                x = self.nifty_loader(path)
                y = self.nifty_loader(y_paths[i])
                
                # Resize to fit input and output size
                #x = cv2.resize(x, (self.dim[1], self.dim[0]))
                #y = cv2.resize(y, (self.dim[1], self.dim[0]))
                
                # Convert y to boolean format
                y = y>(0.5*(np.max(y)-np.min(y))+np.min(y))
                
                # Convert to float32
                y = y.astype("float32")
            else:
                try:
                    # If the input path does not have any suffix try to load it
                    # as a dicom file
                    x = pydicom.dcmread(path)
                    # Get the pixel array
                    x = np.array(x.pixel_array)
                    # Convert to float32
                    x = x.astype(np.float32)
                                      
                    y = pydicom.dcmread(y_paths[i])
                    # Get the pixel array
                    y = np.array(y.pixel_array)
                    # Convert to float32
                    y = y.astype(np.float32)

                    # Resize input and target to fit input and output of the 
                    # network
                    #x = cv2.resize(x, (self.dim[1], self.dim[0]))
                    #y = cv2.resize(y, (self.dim[1], self.dim[0]))
                    
                    # Convert y to boolean form
                    y = y>(0.5*(np.max(y)-np.min(y))+np.min(y))
                    # Convert y to float32
                    y = y.astype("float32")

                except:
                    # Raise error if the file neither have any suffix nor is a
                    # dicom file
                    raise ValueError('File Format is not supported')
            
            
            # Augment the input if the augmentation variable is True
            if augmentation == True:
                x, y = self.augmentor(x, y)
            
            # Normalize the input image
            x = self.transform(x)
            
            # Add channel dimention to the input and target
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)
            
            # Add the second channel if the n_channels is equal to 2
            if self.n_channels == 2:                
                # Add iso filter channel
                p = x > (np.mean(np.array(x))+2*np.std(np.array(x)))
                x_prime = p*x            
                x = np.concatenate((x, x_prime), axis=2)

            # Store input images
            X[i,] = x.astype(np.float32)
            # Store targets
            Y[i,] = y.astype(np.float32)
            

        return X, Y
    














 
