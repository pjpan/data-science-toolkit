#coding=utf8
##########################################################################
# File Name: saved_model_conversion.py
# Author: Meng Zhao
# Created Time: 2020年12月14日 星期一 14时46分47秒
#########################################################################
import os
import sys
import codecs
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

lib_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(lib_path)

import tf_utils
import tool


def convert_saved_model(update_feature_names, input_saved_model_dir, output_saved_model_dir):
    '''
    reduce some features shape
    '''
    update_feature_names = set(update_feature_names)
    model = keras.models.load_model(input_saved_model_dir)
    model.summary()

    #get layer ids
    update_feature_layer_ids = [i for i, elem in enumerate(model.layers) if \
            isinstance(elem, keras.layers.InputLayer) and elem.name in update_feature_names]
   
    # get tile num from anchor feature
    anchor_idx = 0
    for i, item in enumerate(model.layers):
        if i not in update_feature_names:
            anchor_idx = i
            break

    inputs = model.inputs
    
    
    # change input shape and tile them
    new_inputs = []
    for i, item in enumerate(inputs):
        multiplies = tf.one_hot(0, tf.rank(item), dtype=tf.int32) * tf.shape(inputs[anchor_idx])[0] \
                - tf.one_hot(0, tf.rank(item), dtype=tf.int32) \
                + tf.ones(tf.rank(item), dtype=tf.int32)
        if i not in update_feature_layer_ids:
            new_inputs.append(item)
        else:
            new_inputs.append(tf.tile(item, multiplies))
    outputs = model(new_inputs)

    # feed new input to model
    new_model = keras.Model(inputs, outputs)
    new_model.summary()
    new_model.save(output_saved_model_dir) 




def read_user_feature_name_file(input_file):
    '''
    read user feature names 
    '''
    update_feature_names = set()
    with codecs.open(input_file, 'r', 'utf8') as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            line_info = line.split('\t')
            if len(line_info) < 2:
                print('skip line:{}'.format(i))
                continue
            feature_name = line_info[0]
            feature_type = line_info[1]
            update_feature_names.add(feature_name)
    return update_feature_names

