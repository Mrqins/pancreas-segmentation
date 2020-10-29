from __future__ import division, print_function
from keras.layers import Input, Conv2D, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.utils.vis_utils import plot_model
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, ZeroPadding2D, add, GlobalAveragePooling2D, AveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,GlobalAveragePooling2D,multiply, AveragePooling2D
import keras.layers
from tensorflow.contrib.keras import backend as kk
import tensorflow as tf
from sklearn.metrics  import log_loss
from keras.losses import binary_crossentropy
from math import log


from tensorflow.contrib.keras import layers

flt = 32

def dice_coef(y_true, y_pred):
    y_true_f = 2*tf.reduce_sum(y_true*y_pred, axis=(0, 1, 2))
    y_pred_f = tf.reduce_sum(y_pred**2+y_true, axis=(0, 1, 2))
    dcie = (y_true_f+0.0001)/(y_pred_f+0.0001)
    dice = tf.reduce_mean(dcie)
    return dice
def pancrease(y_true, y_pred):
    y_true_f = 2 * tf.reduce_sum(y_true[:, :, :, 1:2] * y_pred[:, :, :, 1:2], axis=(0, 1, 2))
    y_pred_f = tf.reduce_sum(y_pred[:, :, :, 1:2]**2 + y_true[:, :, :, 1:2], axis=(0, 1, 2))

    dcie = (y_true_f + 0.0001) / (y_pred_f + 0.0001)
    dice = tf.reduce_mean(dcie)
    return dice
def cancer(y_true, y_pred):
    y_true_f = 2 * tf.reduce_sum(y_true[:, :, :, 2:3] * y_pred[:, :, :, 2:3], axis=(0, 1, 2))
    y_pred_f = tf.reduce_sum(y_pred[:, :, :, 2:3]**2 + y_true[:, :, :, 2:3], axis=(0, 1, 2))
    dcie = (y_true_f + 0.0001) / (y_pred_f + 0.0001)
    dice = tf.reduce_mean(dcie)
    return dice

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def unet(input_size = (192,256,1)):

    inputs = Input(input_size)

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)

    up6 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef, pancrease, cancer])
    plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=False)

    return model




#attention u-net
def att_unet(input_size = (192,256,1)):
    inputs = Input(input_size)

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv1)
    att1 = conv1 #192*256*64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv2)
    att2 = conv2#96*128*128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv3)
    att3 = conv3#48*64*256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    att4 = conv4 #24*32*512
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)#12*16*512


    up6 = spatial_attention(att4, conv5, 8)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)

    up7 = spatial_attention(att3, conv6, 4)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)

    up8 = spatial_attention(att2, conv7, 2)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = spatial_attention(att1, conv8, 1)
    conv9 = concatenate([up9, Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8)], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.0001), loss=[dice_coef_loss], metrics=[dice_coef])
    plot_model(model, to_file='Attunet_model.png', show_shapes=True, show_layer_names=False)

    return model






def denseblock(in_npy,k):

    dense1 = Conv2D(flt*k, (3, 3), activation='relu', padding='same')(in_npy)
    x1 = dense1
    dense1 = Conv2D(flt*k, (3, 3), activation='relu', padding='same')(dense1)
    x2 = dense1
    dense1 = keras.layers.add([Conv2D(flt*k, (3, 3), activation='relu', padding='same')(dense1), x1])
    dense1 = keras.layers.add([Conv2D(flt*k, (3, 3), activation='relu', padding='same')(dense1), x2])
    dense1 = keras.layers.add([dense1, x1])
    return dense1


def expend_as(tensor, rep):
    return keras.layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)

def spatial_attention(coder_x,up_x,K):

    shape_coder_x = kk.int_shape(coder_x)
    print('coder=',shape_coder_x)
    shape_up_x = kk.int_shape(up_x)


    theta_x = Conv2D(K*flt, (2, 2), padding='same')(coder_x)
    shape_theta_x = kk.int_shape(theta_x)
    print('theta_x=shape:',shape_theta_x)

    phi_g = Conv2D(K*flt, (1, 1), padding='same')(up_x)
    upsample_g = Conv2DTranspose(K*flt, (2, 2), strides=(2, 2))(phi_g)
    print('upsample_g=shape:',upsample_g.shape)

    concat_xg = keras.layers.add([upsample_g, theta_x])
    act_xg = keras.layers.Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = keras.layers.Activation('sigmoid')(psi)


    shape_sigmoid = kk.int_shape(sigmoid_xg)


    upsample_psi = UpSampling2D(size = (shape_coder_x[1]//shape_sigmoid[1], shape_coder_x[2]//shape_sigmoid[2]))(sigmoid_xg)
    print('upsample_psi.shape=',upsample_psi.shape)

    upsample_psi = expend_as(upsample_psi, shape_coder_x[3])


    #upsample_psi = tf.keras.backend.repeat_elements(upsample_psi, shape_coder_x[3], axis=3)


    y = multiply([upsample_psi, theta_x])

    result = Conv2D(shape_coder_x[3], (1, 1), padding='same')(y)
    result_bn = keras.layers.BatchNormalization()(result)
    print('result_bn.shape:'+str(result_bn.shape))
    return result_bn






def cab(batch_input1,batch_input2):

    batch_input = concatenate([batch_input1, batch_input2], axis=3)

    global_avg_pooling = AveragePooling2D(pool_size=(int(batch_input1.shape[1]), int(batch_input1.shape[2])),padding='same')(batch_input)

    batch_input = Conv2D(int(global_avg_pooling.shape[3]), (1, 1), activation='relu', padding='same')(global_avg_pooling)


    batch_input = Conv2D(int(batch_input.shape[3]//2), (1, 1), activation='sigmoid', padding='same')(batch_input)




    mul = multiply([batch_input1, batch_input])


    batch_output = concatenate([batch_input2, mul], axis=3)
    print('batch_output='+str(batch_output.shape))
    return batch_output


def Dense_unet(input_size = (192,256,1)):

    inputs = Input(input_size)

    conv1 = denseblock(inputs, 1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)# 96*128*64

    conv2 = denseblock(pool1, 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #48*64*128

    conv3 = denseblock(pool2, 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #24*32*256

    conv4 = denseblock(pool3, 8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)#12*16*512

    conv5 = Conv2D(flt * 16, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv5)

    up6 = concatenate([Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv6)

    up7 = concatenate([Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv7)

    up8 = concatenate([Conv2DTranspose(flt * 2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv8)

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.00001), loss=[dice_coef_loss], metrics=[dice_coef])
    plot_model(model, to_file='Dense_unet_model.png', show_shapes=True, show_layer_names=False)

    return model

def get_MAunet():
    """build and compile Neural Network"""

    print("start building NN")
    inputs = Input((192, 256, 1))

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(inputs)
    x1 = conv1
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = keras.layers.add([x1, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)#96*128*64
    #muti_level convolution
    conv11 = Conv2D(flt*2, (1, 1), activation='relu', padding='same')(conv1)
    conv12 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv1)
    conv13 = Conv2D(flt*2, (5, 5), activation='relu', padding='same')(conv1)
    conv11 = concatenate([Conv2D(flt * 2, (1, 1), strides=(1, 1), padding='same')(conv12), conv11], axis=3)
    conv11 = concatenate([Conv2D(flt * 2, (1, 1), strides=(1, 1), padding='same')(conv13), conv11], axis=3)
    conv11 = Conv2D(flt, (1, 1), activation='relu', padding='same')(conv11)#192*256*64


    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv2)
    x2 = conv2#192*256*64

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)#96*128*128


    conv3 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv3)
    x3 = conv3#96*128*128
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)#48*64*256


    conv4 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv4)
    x4 = conv4
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    x5 = pool4 #12*16
    conv5 = Conv2D(flt*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = keras.layers.add([x5, conv5])


    #att_6 = spatial_attention(x4, conv5, 8)
    up6 = Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = cab(x4, up6)
    #up6 = concatenate([att_6, up6],axis=3)
    conv6 = denseblock(up6, 8)

    #att_7 = spatial_attention(x3, conv6, 4)
    up7 = Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv6)
    #up7 = concatenate([att_7, up7], axis=3)
    up7 = cab(x3, up7)
    conv7 = denseblock(up7, 4)

    #att_8 = spatial_attention(x2, conv7, 2)
    up8 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = cab(x2, up8)
    #up8 = concatenate([x2, att_8], axis=3)
    conv8 = denseblock(up8, 2)


    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv11], axis=3)
    conv9 = denseblock(up9, 1)


    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.compile(optimizer=Adam(lr=0.00001), loss=(dice_coef_loss+two_log_loss), metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def get_MDAunet():
    """build and compile Neural Network"""

    print("start building NN")
    inputs = Input((192, 256, 1))

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(inputs)
    x1 = conv1
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = keras.layers.add([x1, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #muti_level convolution
    conv11 = Conv2D(flt*2, (1, 1), activation='relu', padding='same')(conv1)
    conv12 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv1)
    conv13 = Conv2D(flt*2, (5, 5), activation='relu', padding='same')(conv1)
    conv11 = concatenate([Conv2D(flt * 2, (1, 1), strides=(1, 1), padding='same')(conv12), conv11], axis=3)
    conv11 = concatenate([Conv2D(flt * 2, (1, 1), strides=(1, 1), padding='same')(conv13), conv11], axis=3)
    conv11 = Conv2D(flt, (1, 1), activation='relu', padding='same')(conv11)



    conv2 = denseblock(pool1, 2)#96*128
    x2 = conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = denseblock(pool2, 4)#48*64
    x3 = conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = denseblock(pool3, 8)#24*32
    x4 = conv4
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    x5 = pool4 #12*16
    conv5 = Conv2D(flt*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv5)

    conv5 = keras.layers.add([x5, conv5])


    #att_6 = spatial_attention(x4, conv5, 8)
    up6 = Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = cab(x4, up6)
    #up6 = concatenate([att_6, up6],axis=3)
    conv6 = denseblock(up6, 8)

    #att_7 = spatial_attention(x3, conv6, 4)
    up7 = Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv6)
    #up7 = concatenate([att_7, up7], axis=3)
    up7 = cab(x3, up7)
    conv7 = denseblock(up7, 4)

    #att_8 = spatial_attention(x2, conv7, 2)
    up8 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = cab(x2, up8)
    conv8 = denseblock(up8, 2)

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv11], axis=3)
    conv9 = denseblock(up9, 1)


    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    #model.compile(optimizer=Adam(lr=0.00001), loss=(dice_coef_loss+two_log_loss), metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coef_loss, metrics=[dice_coef, pancrease, cancer])

    return model



def get_unet():
    """build and compile Neural Network"""

    print("start building NN")
    inputs = Input((192, 256, 1))

    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(inputs)
    x1 = conv1
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = keras.layers.add([x1, conv1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #muti_level convolution
    conv11 = Conv2D(flt*2, (1, 1), activation='relu', padding='same')(conv1)
    conv12 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv1)
    conv13 = Conv2D(flt*2, (5, 5), activation='relu', padding='same')(conv1)
    conv11 = concatenate([Conv2D(flt * 2, (1, 1), strides=(1, 1), padding='same')(conv12), conv11], axis=3)
    conv11 = concatenate([Conv2D(flt * 2, (1, 1), strides=(1, 1), padding='same')(conv13), conv11], axis=3)
    conv11 = Conv2D(flt, (1, 1), activation='relu', padding='same')(conv11)


    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(pool1)
    x2 = conv2
    conv2 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = keras.layers.add([x2, conv2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(pool2)
    x3 = conv3
    conv3 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = keras.layers.add([x3, conv3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(pool3)
    x4 = conv4
    conv4 = Conv2D(flt*8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = keras.layers.add([x4, conv4])
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


    conv5 = Conv2D(flt*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same')(conv5)
    x5 = pool4
    conv5 = keras.layers.add([x5, conv5])


    up6 = concatenate([Conv2DTranspose(flt*8, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(up6)
    x6 = conv6
    conv6 = Conv2D(flt*4, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = keras.layers.add([x6, conv6])

    up7 = concatenate([Conv2DTranspose(flt*4, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(up7)
    x7 = conv7
    conv7 = Conv2D(flt*2, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = keras.layers.add([x7, conv7])

    up8 = concatenate([Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same')(up8)
    x8 = conv8
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = keras.layers.add([x8, conv8])

    up9 = concatenate([Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv8), conv11], axis=3)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(up9)
    x9 = conv9
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = keras.layers.add([x9, conv9])

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def Segnet():
    print("start building NN")
    inputs = Input((192, 256, 1))
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    mask1 = pool1  #96*128

    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(flt * 2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    mask2 = pool2  #48*64

    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    mask3 = pool3  #24*32

    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    mask4 = pool4  #12*16

    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)  #6*8

    up6 = Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(pool5) #12*16
    conv6 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = keras.layers.add([conv6, mask4])

    up7 = Conv2DTranspose(flt * 8, (2, 2), strides=(2, 2), padding='same')(conv6)  #24*32
    conv7 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(flt * 8, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(flt * 4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = keras.layers.add([conv7, mask3])

    up8 = Conv2DTranspose(flt * 4, (2, 2), strides=(2, 2), padding='same')(conv7)  #48*64
    conv8 = Conv2D(flt*4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(flt*4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(flt*2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = keras.layers.add([conv8, mask2])

    up9 = Conv2DTranspose(flt*2, (2, 2), strides=(2, 2), padding='same')(conv8)  #96*128
    conv9 = Conv2D(flt*2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = keras.layers.add([conv9, mask1])

    up10 = Conv2DTranspose(flt, (2, 2), strides=(2, 2), padding='same')(conv9)  #192*256
    conv10 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up10)
    conv10 = Conv2D(flt, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv10)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=0.00001), loss=[dice_coef_loss], metrics=[dice_coef])
    plot_model(model, to_file='unet_model.png', show_shapes=True, show_layer_names=False)

    return model


def train():
    print("loading data done")

    model = get_MDAunet()

    train_image = np.load('/media/fly/B8D46DEEC022AA4B/Qins/pancrease_cancer/imgs_train_0_Z.npy')
    train_mask= np.load('/media/fly/B8D46DEEC022AA4B/Qins/pancrease_cancer/masks_train_0_Z.npy')
    #train_image = train_image[..., np.newaxis]
    print(train_image.shape)
    print(train_mask.shape)
    #train_mask = train_mask[..., np.newaxis]
    train_image = train_image.astype('float32')
    train_mask = train_mask.astype('float32')



    '''
    model = load_model('/media/fly/important/Mrqin/pancreas_seg_NIH_data/models/1e-3MDenseAttunetmodel_2.hdf5', compile=True,custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    model_checkpoint = ModelCheckpoint(filepath='./1e-3MDenseAttunetmodel_2.hdf5', monitor='loss',  save_best_only=True, period=10)
    csv_logger = CSVLogger('./1e-3MDenseAttunetmodel_2.csv')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=30, mode='auto')

    model.fit(train_image, train_mask, batch_size= 16, epochs=300, verbose=1, shuffle=True, callbacks=[model_checkpoint, csv_logger, reduce_lr])
    '''




    #model = load_model('/media/fly/important/Mrqin/pancreas_seg_NIH_data/models/Attunetmodel_3_0.hdf5', compile=True,custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    model_checkpoint = ModelCheckpoint(filepath='./models/get_MAunet_fd0_Z_ep100_lr1e-05.csv.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger('./logs/get_unet_fd0'
                           '_Z_ep100_lr1e-04.csv')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, mode='auto')
    EarlyStop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto' )
    model.fit(train_image, train_mask, batch_size= 16, epochs=100, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, csv_logger, reduce_lr,EarlyStop])

if __name__ == '__main__':
    train()
