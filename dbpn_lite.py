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

def create_non_trainable_model(base_model, BOTTLENECK_TENSOR_NAME, use_global_average = False):
    '''
    Parameters
    ----------
    base_model: This is the pre-trained base model with which the non-trainable model is built

    Note: The term non-trainable can be confusing. The non-trainable-parametes are present only in this
    model. The other model (trianable model doesnt have any non-trainable parameters). But if you chose to 
    omit the bottlenecks due to any reason, you will be training this network only. (If you choose
    --omit_bottleneck flag). So please adjust the place in this function where I have intentionally made 
    certain layers non-trainable.

    Returns
    -------
    non_trainable_model: This is the model object which is the modified version of the base_model that has
    been invoked in the beginning. This can have trainable or non trainable parameters. If bottlenecks are
    created, then this network is completely non trainable, (i.e) this network's output is the bottleneck
    and the network created in the trainable is used for training with bottlenecks as input. If bottlenecks
    arent created, then this network is trained. So please use accordingly.
    '''
    # This post-processing of the deep neural network is to avoid memory errors
    x = (base_model.get_layer(BOTTLENECK_TENSOR_NAME))
    all_layers = base_model.layers
    for i in range(base_model.layers.index(x)):
        all_layers[i].trainable = False
    mid_out = base_model.layers[base_model.layers.index(x)]
    variable_summaries(mid_out.output)
    non_trainable_model = Model(base_model.input, mid_out.output)
    #non_trainable_model = Model(inputs = base_model.input, outputs = [x])
    
    # for layer in non_trainable_model.layers:
    #     layer.trainable = False
    
    return (non_trainable_model)

def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def up_projection(lt_, nf, s, block):
    with tf.name_scope('up_' + str(block)):
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


def down_projection(ht_, nf, s, block, act='prelu' ):
    with tf.name_scope('down_'+str(block)):
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


def super_resolution(input_tensor, n_feature_layers=2, n_projection=1,
            feature_filters=[128, 32], projection_filters=32, k_size=[3, 1], s=2):
    '''
    input_tensor: The input tensor required to complete the model
    n_features_layer: The number of initial feature extraction layers that needs to be 
                    fixed before starting the up and down projection blocks
    n_projection: Number of Up and down projection blocks
    feature_filters: Number of filters in each feature extraction layer
    k_size: The kernel size to be used in feature extraction layer
    s: The scaling factor of the super resolution network

    Returns
    -------
    A Model object of the super resolution network  
    '''
    if not K_B.is_keras_tensor(input_tensor):
        input_tensor = Input(shape=input_tensor.get_shape()[1:], tensor=input_tensor)
    x = Conv2D(feature_filters[0], k_size[0], padding="SAME")(input_tensor)
    if (len(k_size) != len(feature_filters)):
        raise ValueError(
            "Number of Layers for feature extraction must be equal to the number of filters sets given.")
    for i in range(len(k_size)-1):
        x = Conv2D(feature_filters[i+1], k_size[i+1], padding="same")(x)
    for i in range(n_projection):
        x = up_projection(x, projection_filters, s, i+1)
        x = down_projection(x, projection_filters, s, i+1)
    x = up_projection(x, projection_filters, s, n_projection+1)
    x = Conv2D(3, 3, padding='same')(x)
    return Model(inputs=input_tensor, outputs=x) 

def loss_funcs(b,labels):
    out = b.output
    mse = tf.losses.mean_squared_error(out,labels,reduction=tf.losses.Reduction.MEAN)
    
    with tf.name_scope('loss'):
        variable_summaries(mse)
    
    with tf.name_scope('Predictions'):
        variable_summaries(out)
    
    return mse

def perpetual_loss(b,labels):
    '''
    Compute the Perpetual Loss Function based on a VGGNet Trained on ImageNet
    '''
    out = b.output
    print(out)
    print(labels)
    input_tensor = tf.concat([out,labels],axis=0)
    perpetual_model = applications.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    BOTTLENECK_TENSOR_NAME = 'block4_conv3' 
    f = create_non_trainable_model(perpetual_model, BOTTLENECK_TENSOR_NAME)
    vgg_out = f.output
    shape_vgg = tf.shape(vgg_out)
    B = shape_vgg[0]
    H = shape_vgg[1]
    W = shape_vgg[2]
    C = shape_vgg[3]
    feat_recons_loss = tf.losses.mean_squared_error(vgg_out[:B//2,:,:,:],vgg_out[B//2:,:,:,:],reduction=tf.losses.Reduction.MEAN)
    psi_y_pred = tf.reshape(vgg_out[:B//2,:,:,:],[-1,C,H*W])
    gram_y_pred = tf.matmul(psi_y_pred,tf.transpose(psi_y_pred,[0,2,1]))/tf.cast(H*W*C,tf.float32)
    psi_y_tar = tf.reshape(vgg_out[B//2:,:,:,:],[-1,C,H*W])
    gram_y_tar = tf.matmul(psi_y_tar,tf.transpose(psi_y_tar,[0,2,1]))/tf.cast(H*W*C,tf.float32)
    
    style_transfer_loss = tf.reduce_mean(tf.norm(gram_y_pred-gram_y_tar,'fro',axis=(1,2)))  #frobenius norm of gram matrices

    with tf.name_scope('feat_recons_loss'):
        variable_summaries(feat_recons_loss)
    
    with tf.name_scope('Style_transfer_loss'):
        variable_summaries(style_transfer_loss)
    
    return(feat_recons_loss,style_transfer_loss)

if __name__ == '__main__':
    x = Input(shape=(512, 512, 3))
    sample = super_resolution(x)
    print (sample.summary())
