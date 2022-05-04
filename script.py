import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model

from PIL import Image       

# Image/mask file paths
image_path = './data/images/'
mask_path = './data/masks/'

# U-net model input shape
img_height = 96
img_width = 128
num_channels = 3



def show_ground_truth(n):
    '''
    Display nth pair of unprocessed training image and unprocessed ground truth mask.

    Arguments
    ---------
    n: int
        nth training image/mask pair to display
    '''

    img = Image.open(image_list[n])
    mask = Image.open(mask_list[n]).getchannel(0)   # get R channel for intelligible representation of mask

    fig, axs = plt.subplots(1, 2, figsize=(12,12))
    axs[0].imshow(img)
    axs[1].imshow(mask)
    axs[0].set_title('Image')
    axs[1].set_title('Mask')

    # Turn off axes for all subplots
    for i in range(len(axs)):
        axs[i].axis('off')

    plt.show()


def preprocess(image_path, mask_path):
    '''
    Preprocess training dataset consisting of image/mask pairs. Convert image/mask pairs to 
    arrays and resize to img_height, img_width. Flatten masks to monochromatic images.

    Arguments
    ---------
    image_path: string
        file path to training images
    mask_path: string
        file path to training masks

    Returns
    -------
    img: array of shape (height,width,3)
        processed training image
    mask: array of shape (height,width,1)
        processed training mask
    '''

    img = tf.io.read_file(image_path)                             
    img = tf.image.decode_png(img, channels=3)                # img is RGBA; channels=3 discards A channel
    img = tf.image.convert_image_dtype(img, tf.float32)       
    img = tf.image.resize(img, (96, 128), method='nearest')

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)   # take max across channels axis; return monochromatic img
    mask = tf.image.resize(mask, (96, 128), method='nearest')

    return img, mask                                        


def double_conv(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    '''
    Downsample input with successive 2D convolutions. Optionally add dropout and
    max pooling layers.
    
    Arguments
    --------
    inputs: tensor
        Keras Input layer
    n_filters: int
        number of filters in conv layers
    dropout_prob: float
        dropout probability
    max_pooling: bool
        if True, add 2D max pooling layer

    Returns
    -------
    next_layer: tensor
        Keras layer
    skip_layer: tensor
        Keras layer
    '''

    conv = Conv2D(n_filters, 
                  3,   
                  activation="relu",
                  padding="same",
                  kernel_initializer="he_normal")(inputs)
    conv = Conv2D(n_filters, 
                  3,   
                  activation="relu",
                  padding="same",
                  kernel_initializer="he_normal")(conv)
    
    # if dropout_prob > 0 add dropout layer
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)                                
        
    # if max_pooling is True add MaxPooling2D layer
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2,2))(conv)                
    else:
        next_layer = conv
        
    skip_layer = conv
    
    return next_layer, skip_layer


def expand_conv(expand_input, contract_input, n_filters=32):
    '''
    Upsample input with transpose convolution. Merge upsampled input with
    skip layer, and apply successive 2D convolutions.
    
    Arguments
    ---------
    expand_input: tensor
        input tensor from previous layer
    contract_input: tensor
        tensor from previous skip layer
    n_filters: int 
        mumber of filters in conv layers
    
    Returns
    -------
    conv: tensor
        output from merged inputs
    '''
    
    up = Conv2DTranspose(
                 n_filters,    
                 3,    
                 strides=2,
                 padding="same")(expand_input)

    merge = concatenate([up, contract_input], axis=3)

    conv = Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation="relu",
                 padding="same",
                 kernel_initializer="he_normal")(merge)
    conv = Conv2D(n_filters,  # Number of filters
                 3,   # Kernel size
                 activation="relu",
                 padding="same",
                 kernel_initializer="he_normal")(conv)
    
    return conv


def build_model(input_size=(96,128,3), n_filters=32, n_classes=23):
    '''
    Construct U-net model. 
    
    Arguments
    ---------
    input_size: tuple 
        input shape (height,width,channels)
    n_filters: int
        number of filters in conv layers
    n_classes: int
        number of output classes for segmentation
    
    Returns
    -------
    model: Keras model      
        U-net model instance
    '''

    inputs = Input(input_size)

    # Downsampling stage (encoding)
    # output of double_conv - dblock1/dblock2 - is a tuple: (next_layer, skip_layer)
    dblock1 = double_conv(inputs=inputs, n_filters=n_filters)
    dblock2 = double_conv(dblock1[0], n_filters=n_filters*2)
    dblock3 = double_conv(dblock2[0], n_filters=n_filters*4)
    dblock4 = double_conv(dblock3[0], n_filters=n_filters*8, dropout_prob=0.3) 
    dblock5 = double_conv(dblock4[0], n_filters=n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # Upsampling stage (decoding)
    ublock6 = expand_conv(dblock5[0], dblock4[1], n_filters=n_filters*8)
    ublock7 = expand_conv(ublock6, dblock3[1], n_filters=n_filters*4)
    ublock8 = expand_conv(ublock7, dblock2[1], n_filters=n_filters*2)
    ublock9 = expand_conv(ublock8, dblock1[1], n_filters=n_filters)

    conv10 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv11 = Conv2D(n_classes, 1, padding="same")(conv10)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv11)

    return model

def train_model(dataset, epochs=40):
    '''
    Train U-net model on preprocessed dataset.

    Arguments
    ---------
    dataset: tf.data.Dataset
        preprocessed dataset comprising image/mask pairs
    epochs: int
        number of training epochs
    '''

    model = build_model((img_height, img_width, num_channels))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)
    history = model.fit(dataset, epochs=epochs)
    
    # Plot accuracy
    plt.plot(history.history["accuracy"])

    # Save trained U-net model
    model.save('model/')


def show_processed_images(dataset):
    '''
    Display pair of processed training image and processed ground truth mask.

    Arguments
    ---------
    dataset: tf.data.Dataset
        preprocessed dataset batch
    '''

    for img, mask in dataset:
        display_img, display_mask = img, mask

    plt.figure(figsize=(12,12))

    fig, axs = plt.subplots(1, 2, figsize=(12,12))
    axs[0].imshow(display_img)
    axs[1].imshow(display_mask)
    axs[0].set_title('Image')
    axs[1].set_title('Mask')

    plt.show()


def format_mask(pred_mask):
    '''
    Format predicted mask from trained U-net.

    Arguments
    ---------
    pred_mask: tensor of shape (32, 96, 128, 23) - (batch_size, height_pix, width_pix, class_probs)
        unformatted predicted mask output from U-net

    Returns
    -------
    pred_mask[0]: tensor of shape (96, 128, 1)
        formatted predicted mask output from U-net

    '''

    # argmax() selects max class prob resulting in output of shape (32, 96, 128)
    pred_mask = tf.argmax(pred_mask, axis=-1)

    # add channels axis so mask is proper shape for display
    pred_mask = pred_mask[..., tf.newaxis]

    # pred_mask[0] selects first mask in batch of masks resulting in output of shape (96, 128, 1)
    return pred_mask[0]


def show_predictions(dataset, num=1):
    '''
    Displays first image of each of num batches.

    Arguments
    ---------
    dataset: tf.data.Datset
        preprocessed dataset comprising image/mask pairs
    num: int
        number of batches for which to show image/mask pairs
    '''

    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)

    # take() returns a batch of examples. Selecting the zeroth index of image and mask returns
    # the first image and mask respectively in the batch returned by take().
    for image, mask in dataset.take(num):

        pred_mask = model.predict(image)
        pred_mask = format_mask(pred_mask)

        fig, axs = plt.subplots(1, 3, figsize=(12,12))
        axs[0].imshow(image[0])
        axs[1].imshow(mask[0])
        axs[2].imshow(pred_mask)

        axs[0].set_title('Input image')
        axs[1].set_title('Ground truth mask')
        axs[2].set_title('Predicted mask')

        # Turn off axes for all subplots
        for i in range(len(axs)):
            axs[i].axis('off')

        plt.show()



############################
########---Script---########
############################

buffer_size = 500
batch_size = 32

image_list = os.listdir(image_path)                    
mask_list = os.listdir(mask_path)                      
image_list = [image_path + i for i in image_list]
mask_list = [mask_path + i for i in mask_list]

# Display example unprocessed image/mask pair from training set
show_ground_truth(1)

# Combine two tensors (image tensor, mask tensor) into single Dataset object
dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))

# Preprocess training set
processed_dataset = dataset.map(preprocess)

# Load pre-trained U-net model
model = load_model('unet_model/')
print(model.summary())

# Show processed image/mask pair and corresponding mask predicted by U-net
show_predictions(processed_dataset, 1)





