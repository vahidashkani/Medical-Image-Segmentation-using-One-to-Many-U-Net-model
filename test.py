
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from timeit import default_timer as timer
from datetime import timedelta
from PIL import Image
from PIL import ImageOps


def pil_loader(path):
    """JPG and PNG format loader"""
    img = Image.open(path).convert('L')
    expected_size=[480,320]
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    im = ImageOps.expand(img, padding)
    return im


def transform_(X):
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


# Create a result directory if it is not existed
if not os.path.exists('AC_unet_single_image/'):
    os.makedirs('AC_unet_single_image/')
# Generators
validation_generator = "single_val.txt"
with open(validation_generator, 'r') as f:
    val_list_paths = f.readlines()
fg_val = len(val_list_paths)
indexes_val = np.arange(fg_val)
# Find list of x paths
X_paths_val = [val_list_paths[e] for e in indexes_val]
y_paths_val = [e.replace("\n","").split(" ")[-1] for e in X_paths_val]
x_paths_val = [e.replace("\n","").split(" ")[0] for e in X_paths_val]

#dim = (320,480)
batch_size = 1
n_channels = 1
# = np.empty((batch_size, *dim, n_channels))
#Y_val = np.empty((batch_size, *dim, 1))


start = timer() 
model = keras.models.load_model("AC_unet.h5")
#evaluate validation data
for j, path_val in enumerate(x_paths_val):
    X_vali=[]
    Y_vali=[]
    # Load and preprocess image according to its format
    # it accepts .png, .jpg, .PNG and .JPG suffixes
    if path_val.endswith(('.png', '.PNG', 'jpg', '.JPG')):
        
        x_val = pil_loader(path_val)
        x_val = np.array(x_val) 
        x_val = x_val.astype("float32")

        y_val = pil_loader(y_paths_val[j])
        y_val = np.array(y_val) 
        # Convert y to boolean format
        y_val = y_val>(0.5*(np.max(y_val)-np.min(y_val))+np.min(y_val))
        # Convert to float32
        y_val = y_val.astype("float32")
        # Normalize the input image
        x_val = transform_(x_val)
        # Add channel dimention to the input and target
        c_val = np.expand_dims(x_val, -1)
        s_val = np.expand_dims(y_val, -1)
        X_vali.append(c_val)
        Y_vali.append(s_val)
        X_val=np.array(X_vali)
        Y_val=np.array(Y_vali)

        y = model.predict(X_val)
        y = y>=0.5
        y = np.squeeze(y)
            
        # Remove extra dimensions in groound truth and input image
        y_gt = np.squeeze(s_val)
        image = np.squeeze(c_val)
            
        # Plot the results
        fig = plt.figure(frameon=False)
        fig.set_size_inches(15, 15)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.imshow(image, cmap='gray')
        ax.contour(y_gt, colors='green', linestyles='dotted')
        ax.contour(y, colors='red', linestyles='dotted')
        end = timer()
        #perf_time =  end-start
        
            
        # Save the results in the results directory    
        fig.savefig("AC_unet_single_image/{:03d}.jpg".format(j))
            
print(timedelta(seconds=end-start))
    
    
    
    
    
