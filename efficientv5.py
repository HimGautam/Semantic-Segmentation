# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
print(f" Number of Visible GPU is {len(physical_devices)} ")

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Growth Allowed")
except:
    print("No Thank You")

pass
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D , BatchNormalization 
from tensorflow.keras.layers import Dropout, Activation, Conv2DTranspose
from tensorflow.keras.applications import EfficientNetB4


print("STEP 1")
def get_model(num_class):
    
    
    inputs = Input(shape = (288, 512, 3), name = "input_image")
    encoder = EfficientNetB4(input_tensor = inputs, include_top = False, weights = "imagenet")
    
    encoder.trainable = False
        
    encoder_output = encoder.get_layer("block3d_add").output
    
    
    x =  Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(encoder_output)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x =  Conv2D(480, (3, 3), dilation_rate= 2, padding = "same", kernel_initializer= "he_normal", name = "Depthwise_conv_0")(x)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x1 = tf.reduce_mean(x, axis = [1, 2], keepdims= True, name = "global_avg_pooling_1")
    x1 = Conv2D(28, (1, 1), padding = "same" , kernel_initializer= "he_normal")(x1)
    x1 = Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(x1)
    x =  tf.math.multiply(x, x1, name = "block_excite")
    x =  Conv2D(112, (1, 1), padding = "same", kernel_initializer= "he_normal")(x)
    x  = BatchNormalization()(x)
    y0 = Dropout(0.1)(x)
    
    
    
    x =  Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(y0)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x =  Conv2D(480, (3, 3), dilation_rate= 2, padding = "same", kernel_initializer= "he_normal", name = "Depthwise_conv_1")(x)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x1 = tf.reduce_mean(x, axis = [1, 2], keepdims= True, name = "global_avg_pooling_1")
    x1 = Conv2D(28, (1, 1), padding = "same" , kernel_initializer= "he_normal")(x1)
    x1 = Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(x1)
    x =  tf.math.multiply(x, x1, name = "block_excite")
    x =  Conv2D(112, (1, 1), padding = "same", kernel_initializer= "he_normal")(x)
    x  = BatchNormalization()(x)
    y1 = Dropout(0.1)(x)
    
    y1 = tf.keras.layers.Add()([y0, y1])
    
    x =  Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(y1)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x =  Conv2D(480, (3, 3), dilation_rate= 2, padding = "same", kernel_initializer= "he_normal", name = "Depthwise_conv_2")(x)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x1 = tf.reduce_mean(x, axis = [1, 2], keepdims= True, name = "global_avg_pooling_2")
    x1 = Conv2D(28, (1, 1), padding = "same" , kernel_initializer= "he_normal")(x1)
    x1 = Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(x1)
    x =  tf.math.multiply(x, x1, name = "block_excite")
    x =  Conv2D(112, (1, 1), padding = "same", kernel_initializer= "he_normal")(x)
    x  = BatchNormalization()(x)
    y2 = Dropout(0.1)(x)
    
    y2 = tf.keras.layers.Add()([y1, y2])
    
    x =  Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(y2)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x =  Conv2D(480, (3, 3), dilation_rate= 2, padding = "same", kernel_initializer= "he_normal", name = "Depthwise_conv_3")(x)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x1 = tf.reduce_mean(x, axis = [1, 2], keepdims= True, name = "global_avg_pooling_3")
    x1 = Conv2D(28, (1, 1), padding = "same" , kernel_initializer= "he_normal")(x1)
    x1 = Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(x1)
    x =  tf.math.multiply(x, x1, name = "block_excite")
    x =  Conv2D(112, (1, 1), padding = "same", kernel_initializer= "he_normal")(x)
    x  = BatchNormalization()(x)
    y3 = Dropout(0.1)(x)
    
    
    y3 = tf.keras.layers.Add()([y2, y3])
    
    
    
    x =  Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(y3)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x =  Conv2D(480, (3, 3), dilation_rate= 2, padding = "same", kernel_initializer= "he_normal", name = "Depthwise_conv_4")(x)
    x =  BatchNormalization()(x)
    x =  tf.nn.relu6(x)
    x1 = tf.reduce_mean(x, axis = [1, 2], keepdims= True, name = "global_avg_pooling_4")
    x1 = Conv2D(28, (1, 1), padding = "same" , kernel_initializer= "he_normal")(x1)
    x1 = Conv2D(480, (1, 1), padding = "same", kernel_initializer= "he_normal")(x1)
    x =  tf.math.multiply(x, x1, name = "block_excite")
    x =  Conv2D(112, (1, 1), padding = "same", kernel_initializer= "he_normal")(x)
    x  = BatchNormalization()(x)
    y4 = Dropout(0.1)(x)
    
    
    
    
    y4 = tf.keras.layers.Add()([y3, y4])
    
    
  
    

    
    
    
    
    
    # decoder network

    x = Conv2D(256, (3, 3), padding = "same", kernel_initializer= "he_normal")(y4)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)
    
    y = Conv2DTranspose(256, (2, 2), strides = 2 , kernel_initializer = "he_normal")(x)
    
    
#     skip_connection = encoder.get_layer("block3a_expand_activation").output
    
#     x = tf.concat([x, skip_connection], axis = 3)
    
    
    x = Conv2D(256, (3, 3), padding = "same", kernel_initializer= "he_normal")(y)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)
    
    x = Dropout(0.1)(x)
      
    x = Conv2D(256, (3, 3), padding = "same", kernel_initializer= "he_normal")(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)
    
    y = tf.keras.layers.Add()([x, y])
    
    y = Conv2DTranspose(128, (2, 2), strides = 2 , kernel_initializer = "he_normal")(x)
    
    
    skip_connection = encoder.get_layer("block2a_expand_activation").output
    y = tf.concat([y, skip_connection], axis = 3)
    
    

    x = Conv2D(128, (3, 3), padding = "same", kernel_initializer = "he_normal")(y)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)
    
    x = Dropout(0.1)(x)
    
    x = Conv2D(128, (3, 3), padding = "same", kernel_initializer= "he_normal")(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)
    
    
    y = Conv2DTranspose(64, (2, 2), strides = 2 , kernel_initializer = "he_normal")(y)
    
    
    x = Conv2D(64, (3, 3), padding = "same", kernel_initializer = "he_normal")(y)
    x = Conv2D(num_class, (1, 1), padding = "same", kernel_initializer = "he_normal")(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)
    
    
    x = Conv2D(num_class, (3, 3), padding = "same", kernel_initializer= "he_normal")(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu6(x)
    
    
    x = Conv2D(num_class, (3, 3), padding = "same" )(x)
    
    model  = Model(inputs= inputs, outputs = x)
        
        
        

    
    return model

print("AFTER MODEL")

if __name__ == "__main__":
    print("STEP2")
    NUM_CLASSES = 21
    deeplab = get_model(num_class = NUM_CLASSES)
    deeplab.summary()

