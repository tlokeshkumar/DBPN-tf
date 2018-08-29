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

def up_projection(lt_, nf, s):
    
    if s == 2:
        ht = Conv2DTranspose(nf, 2, strides=2)(lt_)
        ht = PReLU()(ht)
        lt = ZeroPadding2D(2)(ht)       
        lt = Conv2D(nf, 6, 2)(lt)
        lt = PReLU()(lt)
        et = Subtract()([lt, lt_])
        ht1= Conv2DTranspose(nf, 2, strides=2)(et)
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
        ht1= Conv2DTranspose(nf, 4, strides=4)(et)
        ht1 = PReLU()(ht1)
        ht1= Add()([ht, ht1])
        return (ht1)
    if s == 8:
        ht = Conv2DTranspose(nf, 8, strides=8)(lt_)
        ht = PReLU()(ht)
        lt = ZeroPadding2D(2)(ht)
        lt = Conv2D(nf, 12, strides=8)(lt)
        lt = PReLU()(lt)
        et = Subtract()([lt, lt_])
        ht1= Conv2DTranspose(nf, 8, strides=8)(et)
        ht1 = PReLU()(ht1)
        ht1= Add()([ht, ht1])
        return (ht1)

def down_projection(ht_, nf, s, act='prelu'):

    if s == 2:
        ht = ZeroPadding2D(2)(ht_)
        lt = Conv2D(nf, 6 ,strides=2)(ht)
        lt = PReLU()(lt)
        ht = Conv2DTranspose(nf, 2 ,strides=2)(lt)
        ht = PReLU()(ht)
        et = Subtract()([ht, ht_])
        lt1 = ZeroPadding2D(2)(et)
        lt1 = Conv2D(nf, 6, strides=2)(lt1)
        lt1 = PReLU()(lt1)
        lt1 = Add()([lt1, lt])
        return lt1
    if s == 4:
        ht = ZeroPadding2D(2)(ht_)
        lt = Conv2D(nf, 8 ,strides=4)(ht)
        lt = PReLU()(lt)
        ht = Conv2DTranspose(nf, 4 ,strides=4)(lt)
        ht = PReLU()(ht)
        et = Subtract()([ht, ht_])
        lt1 = ZeroPadding2D(2)(et)
        lt1 = Conv2D(nf, 8, strides=4)(lt1)
        lt1 = PReLU()(lt1)
        lt1 = Add()([lt1, lt])
        return lt1
    if s == 8:
        ht = ZeroPadding2D(2)(ht_)
        lt = Conv2D(nf, 12 ,strides=8)(ht)
        lt = PReLU()(lt)
        ht = Conv2DTranspose(nf, 8 ,strides=8)(lt)
        ht = PReLU()(ht)
        et = Subtract()([ht, ht_])
        lt1 = ZeroPadding2D(2)(et)
        lt1 = Conv2D(nf, 12, strides=8)(lt1)
        lt1 = PReLU()(lt1)
        lt1 = Add()([lt1, lt])
        return lt1

if __name__ == '__main__':
    x = Input(shape=(104,104, 32))
    sample = up_projection(x, 32, 2)
    sample = Model(inputs=x, outputs=(sample))
    print (sample.summary())