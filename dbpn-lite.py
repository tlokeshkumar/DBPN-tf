import tensorflow as tf
from tensorflow.python.keras import layers
# from keras.layers import *
# from keras.layers import Conv2DTranspose

from tensorflow.python.keras.layers import (Activation, AveragePooling2D,
                                            BatchNormalization, Conv2D, Conv3D,
                                            Dense, Flatten,
                                            GlobalAveragePooling2D,
                                            GlobalMaxPooling2D, Input,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, Dropout, concatenate,
                                            Conv2DTranspose, ZeroPadding2D,
                                            Subtract, Add, PReLU)

from tensorflow.python.keras import applications
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K_B


def up_projection(lt_, nf, s, block):

    if s == 2:
        ht = Conv2DTranspose(nf, 2, strides=2)(lt_)
        ht = PReLU()(ht)
        lt = ZeroPadding2D(2)(ht)
        lt = Conv2D(nf, 6, 2)(lt)
        lt = PReLU()(lt)
        et = Subtract()([lt, lt_])
        ht1 = Conv2DTranspose(nf, 2, strides=2)(et)
        ht1 = PReLU()(ht1)
        ht1 = Add()([ht, ht1])
        return (ht1)
    if s == 4:
        ht = Conv2DTranspose(nf, 4, strides=4)(lt_)
        ht = PReLU()(ht)
        lt = ZeroPadding2D(2)(ht)
        lt = Conv2D(nf, 8, strides=4)(lt)
        lt = PReLU()(lt)
        et = Subtract()([lt, lt_])
        ht1 = Conv2DTranspose(nf, 4, strides=4)(et)
        ht1 = PReLU()(ht1)
        ht1 = Add()([ht, ht1])
        return (ht1)
    if s == 8:
        ht = Conv2DTranspose(nf, 8, strides=8)(lt_)
        ht = PReLU()(ht)
        lt = ZeroPadding2D(2)(ht)
        lt = Conv2D(nf, 12, strides=8)(lt)
        lt = PReLU()(lt)
        et = Subtract()([lt, lt_])
        ht1 = Conv2DTranspose(nf, 8, strides=8)(et)
        ht1 = PReLU()(ht1)
        ht1 = Add()([ht, ht1])
        return (ht1)


def down_projection(ht_, nf, s, act='prelu'):

    if s == 2:
        ht = ZeroPadding2D(2)(ht_)
        lt = Conv2D(nf, 6, strides=2)(ht)
        lt = PReLU()(lt)
        ht = Conv2DTranspose(nf, 2, strides=2)(lt)
        ht = PReLU()(ht)
        et = Subtract()([ht, ht_])
        lt1 = ZeroPadding2D(2)(et)
        lt1 = Conv2D(nf, 6, strides=2)(lt1)
        lt1 = PReLU()(lt1)
        lt1 = Add()([lt1, lt])
        return lt1
    if s == 4:
        ht = ZeroPadding2D(2)(ht_)
        lt = Conv2D(nf, 8, strides=4)(ht)
        lt = PReLU()(lt)
        ht = Conv2DTranspose(nf, 4, strides=4)(lt)
        ht = PReLU()(ht)
        et = Subtract()([ht, ht_])
        lt1 = ZeroPadding2D(2)(et)
        lt1 = Conv2D(nf, 8, strides=4)(lt1)
        lt1 = PReLU()(lt1)
        lt1 = Add()([lt1, lt])
        return lt1
    if s == 8:
        ht = ZeroPadding2D(2)(ht_)
        lt = Conv2D(nf, 12, strides=8)(ht)
        lt = PReLU()(lt)
        ht = Conv2DTranspose(nf, 8, strides=8)(lt)
        ht = PReLU()(ht)
        et = Subtract()([ht, ht_])
        lt1 = ZeroPadding2D(2)(et)
        lt1 = Conv2D(nf, 12, strides=8)(lt1)
        lt1 = PReLU()(lt1)
        lt1 = Add()([lt1, lt])
        return lt1


def super_resolution(input_tensor, n_feature_layers=2, n_projection=8, feature_filters=[128, 32], projection_filters=32, k_size=[3, 1], s=4):

    x = Conv2D(feature_filters[0], k_size[0], padding="SAME")(input_tensor)
    if (len(k_size) != len(feature_filters)):
        raise ValueError(
            "Number of Layers for feature extraction must be equal to the number of filters sets given.")
    for i in range(len(k_size)-1):
        x = Conv2D(feature_filters[i+1], k_size[i+1], padding="same")(x)
    for i in range(n_projection):
        x = up_projection(x, projection_filters, s)
        x = down_projection(x, projection_filters, s)
    x = up_projection(x, projection_filters, s)
    return Model(inputs=input_tensor, outputs=x)


if __name__ == '__main__':
    x = Input(shape=(104, 104, 32))
    sample = super_resolution(x)
    print(sample.summary())
