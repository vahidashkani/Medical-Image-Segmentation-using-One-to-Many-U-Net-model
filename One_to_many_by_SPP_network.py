
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:24:36 2020

@author: vahid
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, Concatenate)
from tensorflow.keras import optimizers as opt
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, GlobalAveragePooling1D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add, multiply
from keras.layers import Concatenate, UpSampling2D, AveragePooling2D

###############################################################################################################
#define a function in order to calculate the dice_coefficients
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor)

# define a function in order to calculate the jaccard_coefficients
def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

#define a function in order to calculate the loss in dice_coefficients
def loss_dice_coefficient_error(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


###############################################################################################################
# define an encoder block
def define_encoder_block(layer_in, n_filters):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # leaky relu activation
    g = Activation('relu')(g)
    #g = LeakyReLU(alpha=0.2)(g)
    g = BatchNormalization()(g, training=True)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    #g = BatchNormalization()(g, training=True)
    # relu activation
    g = Activation('relu')(g)
    #g = LeakyReLU(alpha=0.2)(g)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    g = BatchNormalization()(g, training=True)
    return g

# generator a resnet block
def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first layer convolutional layer
	g = Conv2D(n_filters, (5,5), padding='same', kernel_initializer=init)(input_layer)
	g = BatchNormalization()(g, training=True)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (7,7), padding='same', kernel_initializer=init)(g)
	g = BatchNormalization()(g, training=True)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

def ASPP(x, filter1, filter2, filter3, filter4):
    shape = x.shape

    y2 = Conv2D(filter1, (3,3), dilation_rate=2, padding="same", use_bias=False)(x)
    #y2 = LeakyReLU(alpha=0.2)(y2)
    y2 = Activation('relu')(y2)
    y2 = BatchNormalization()(y2)

    y3 = Conv2D(filter2, (3,3), dilation_rate=4, padding="same", use_bias=False)(x)
    y3 = Activation("relu")(y3)
    #y3 = LeakyReLU(alpha=0.2)(y3)
    y3 = BatchNormalization()(y3)
    

    y4 = Conv2D(filter3, (3,3), dilation_rate=6, padding="same", use_bias=False)(x)
    #y4 = LeakyReLU(alpha=0.2)(y4)
    y4 = Activation('relu')(y4)
    y4 = BatchNormalization()(y4)
    

    y5 = Conv2D(filter4, (3,3), dilation_rate=8, padding="same", use_bias=False)(x)
    #y5 = LeakyReLU(alpha=0.2)(y5)
    y5 = Activation('relu')(y5)
    y5 = BatchNormalization()(y5)
    

    y = Concatenate()([y2, y3, y4, y5])


    #y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    #y = BatchNormalization()(y)
    #y = Activation("relu")(y)

    return y


def many_to_one(filters, x,x2,x3):
    """Define the attention blocks"""
    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    x3 = Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(x3)
    x3 = Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(x3)
    x3 = Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(x3)
    x3 = Activation('relu')(x3)
    #x3 = LeakyReLU(alpha=0.2)(x3)

    x2 = Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(x2)
    x2 = Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(x2)
    x2 = Activation('relu')(x2)
    #x2 = LeakyReLU(alpha=0.2)(x2)
    g1 = Concatenate()([x3, x2])
    
    x = Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(x)
    x = Activation('relu')(x)
    #x = LeakyReLU(alpha=0.2)(x)
    g2 = Concatenate()([g1, x])    
    
    return g2

###############################################################################################################
# define proposed Unet mode known as Fast Unet in below function
def create_model(input_image_size, n_labels=1,
                        mode='classification',
                        init_lr=0.0001, n_resnet=2):
    
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=input_image_size)
    # encoder model
    e1 = define_encoder_block(in_image, 20)
    #you can get the layers out-puts in NN in below way
    s1 = e1.get_shape()
    print('s1: ',s1)
    e2 = define_encoder_block(e1, 40)
    s2 = e2.get_shape()
    print('s2: ',s2)
    e3 = define_encoder_block(e2, 80)
    s3 = e3.get_shape()
    print('s3: ',s3)
    
    e11 = define_encoder_block(e1, 40)
    s11 = e11.get_shape()
    print('s11: ',s11)
    e12 = define_encoder_block(e11, 80)
    s12 = e12.get_shape()
    print('s12: ',s12)
    e13 = define_encoder_block(e12, 160)
    s13 = e13.get_shape()
    print('s13: ',s13)
    
    e21 = define_encoder_block(e2, 80)
    s21 = e21.get_shape()
    print('s21: ',s21)
    e22 = define_encoder_block(e21, 160)
    s22 = e22.get_shape()
    print('s22: ',s22)
    e23 = define_encoder_block(e22, 320)
    s23 = e23.get_shape()
    print('s23: ',s23)
    
    e31 = define_encoder_block(e3, 160)
    s31 = e31.get_shape()
    print('s31: ',s31)
    e32 = define_encoder_block(e31, 320)
    s32 = e32.get_shape()
    print('s32: ',s32)
    e33 = define_encoder_block(e32, 640)
    s33 = e33.get_shape()
    print('s33: ',s33)
    
    #e4 = define_encoder_block(e3, 40)
    #e5 = define_encoder_block(e4, 320)
    # bottleneck, no batch norm and relu
    b = Conv2D(640, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(e33)
    b = Activation('relu')(b)
  
    bb = b.get_shape()
    print('b: ',bb)
    # decoder model
    d33 = decoder_block(b, e33, 640)
    sd33 = d33.get_shape()
    print('sd33: ',sd33)
    d32 = decoder_block(d33, e32, 320)
    sd32 = d32.get_shape()
    print('sd32: ',sd32)
    d31 = decoder_block(d32, e31, 160)
    sd31 = d31.get_shape()
    print('sd31: ',sd31)
    
    # decoder model
    d23 = decoder_block(d33, e23, 320)
    sd23 = d23.get_shape()
    print('sd23: ',sd23)
    d22 = decoder_block(d23, e22, 160)
    sd22 = d22.get_shape()
    print('sd22: ',sd22)
    d21 = decoder_block(d22, e21, 80)
    sd21 = d21.get_shape()
    print('sd21: ',sd21)
    
    d13 = decoder_block(d32, e13, 160)
    sd13 = d13.get_shape()
    print('sd13: ',sd13)
    d12 = decoder_block(d13, e12, 80)
    sd12 = d12.get_shape()
    print('sd12: ',sd12)
    d11 = decoder_block(d12, e11, 40)
    sd11 = d11.get_shape()
    print('sd11: ',sd11)
    
    d3 = decoder_block(d31, e3, 80)
    sd3 = d3.get_shape()
    print('sd3: ',sd3)
    d2 = decoder_block(d3, e2, 40)
    sd2 = d2.get_shape()
    print('sd2: ',sd2)
    d1 = decoder_block(d2, e1, 20)
    sd1 = d1.get_shape()
    print('sd1: ',sd1)
    
    g = many_to_one(20, d1, d11, d21) 
    g = ASPP(g, 20, 40, 80, 160)

    g = Conv2DTranspose(1, (7,7), strides=(1,1), padding='same', kernel_initializer=init)(g)
    out_image = Activation('sigmoid')(g)
    # define model
    unet_model = Model(in_image, out_image)
    number_of_classification_labels = n_labels
    if number_of_classification_labels == 1:
        unet_model.compile(loss=loss_dice_coefficient_error, 
                                optimizer=opt.Adam(lr=init_lr), metrics=[dice_coefficient, jaccard_coef])
    return unet_model




