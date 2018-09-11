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
from input_utils import read_no_labels
from dbpn_lite import super_resolution, loss_funcs
import argparse
import numpy as np
import coloredlogs
import os


parser = argparse.ArgumentParser(description="Inputs to the code")

parser.add_argument("--train_dir",type=str,help="path to directory with training examples")
parser.add_argument("--batch_size",type=int,default=16,help="Batch Size")
parser.add_argument("--log_directory",type = str,default='./log_dir',help="path to tensorboard log")
parser.add_argument("--ckpt_savedir",type = str,default='./checkpoints/model_ckpt',help="path to save checkpoints")
parser.add_argument("--load_ckpt",type = str,default='./checkpoints',help="path to load checkpoints from")
parser.add_argument("--save_freq",type = int,default=100,help="save frequency")
parser.add_argument("--display_step",type = int,default=1,help="display frequency")
parser.add_argument("--summary_freq",type = int,default=100,help="summary writer frequency")
parser.add_argument("--no_epochs",type=int,default=10,help="number of epochs for training")

args = parser.parse_args()
no_iter_per_epoch = np.ceil(30000/args.batch_size)

runopts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
coloredlogs.install(level='DEBUG')
tf.logging.set_verbosity(tf.logging.DEBUG)

init = tf.global_variables_initializer()

n, ini = read_no_labels(args.train_dir, s=8, patch=32, batch_size=args.batch_size)

sample = super_resolution(n[1], s=8, n_projection=8)

loss = loss_funcs(sample, n[0])

global_step_tensor = tf.train.get_or_create_global_step()
init_learning_rate = tf.constant(1e-4)
learning_rate = tf.train.exponential_decay(init_learning_rate,global_step_tensor,decay_rate=0.1,decay_steps=5e5,staircase=True)
tf.summary.image('High-Res-True', n[0])
tf.summary.image('High-Res-Pred', sample.output)
tf.summary.image('Low-Res', n[1])

optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=1e-4)
opA = optimizer.minimize(loss,global_step=global_step_tensor)

with K_B.get_session() as sess:
        
    sess.run(init)
    # initialize iterations variables
    sess.run(ini)
    summary_writer = tf.summary.FileWriter(args.log_directory, sess.graph)    
    summary = tf.summary.merge_all()
    
    saver = tf.train.Saver()

    tf.logging.info('Tensorboard logs will be written to ' + str(args.log_directory))

    if args.load_ckpt is not None:

        if os.path.exists(args.load_ckpt):
            if tf.train.latest_checkpoint(args.load_ckpt) is not None:
                tf.logging.info('Loading Checkpoint from '+ tf.train.latest_checkpoint(args.load_ckpt))
                saver.restore(sess, tf.train.latest_checkpoint(args.load_ckpt))

            else:
                tf.logging.info('Training from Scratch -  No Checkpoint found')
    
    else:
        tf.logging.info('Training from scratch')

    tf.logging.info('Training with Batch Size %d for %d epochs'%(args.batch_size,args.no_epochs))

    while True:    
    # Training Iterations Begin
        global_step,_ = sess.run([global_step_tensor,opA],options = runopts)
        
        if global_step%(args.display_step)==0:
            loss_val = sess.run([loss],options = runopts)
            tf.logging.info('Iteration: ' + str(global_step) + ' Loss: ' +str(loss_val))
        
        if global_step%(args.summary_freq)==0:
            tf.logging.info('Summary Written')
            summary_str = sess.run(summary)
            summary_writer.add_summary(summary_str, global_step)
        
        if global_step%(args.save_freq)==0:
            saver.save(sess,args.ckpt_savedir,global_step=tf.train.get_global_step())
        
        if np.floor(global_step/no_iter_per_epoch) == args.no_epochs:
            break        

