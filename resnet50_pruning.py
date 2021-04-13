from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
from resnet50 import create_model
from solver import create_admm_solver
from TFRecord_data_gen import data_iterator
from prune_utility import apply_prune_on_grads,apply_prune
import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
from numpy import linalg as LA
from pattern_select import get_patterns
from connectivity_pruned import pruning_bad_connect
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

FLAGS = None

input_channel = 3
input_width = 224
input_height = 224

#input preprocess
mean = 128
scale = 1.0/128


#------------------训练集------------------#
with open('../../TRAIN_DATA/imagenet_train.txt','r') as f:
  imglist = f.readlines()
  np.random.shuffle(imglist)
  img_num = len(imglist)

def data_generator(batch_size=128):
    current_index = 0
    while True:
        X = []
        Y = []

        if current_index+batch_size>img_num:
            print('shuffle the train data!')
            np.random.shuffle(imglist)
            current_index = 0

        for index in range(batch_size):
          imagepath, label = imglist[current_index+index].split()

          #input
          img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
          img_resize = cv2.resize(img, (input_width, input_height))
          img_data = (img_resize.astype(np.float32) - mean) * scale
          X.append(img_data)

          #label
          output_data = np.zeros(1000)
          output_data[int(label)] = 1.0
          Y.append(output_data)

        current_index += batch_size

        X = np.array(X)
        Y = np.array(Y)
        yield { 'inputs': X.reshape((batch_size,input_height,input_width,input_channel)), 'labels': Y.reshape((batch_size, 1000)) }      


#------------------验证集------------------#
with open('../../TRAIN_DATA/imagenet_val.txt','r') as f:
  imglist_val = f.readlines()
  np.random.shuffle(imglist_val)
  img_num_val = len(imglist_val)

def data_generator_val(batch_size=128):
    current_index = 0
    while True:
        X = []
        Y = []

        if current_index+batch_size>img_num_val:
          print('shuffle the val data!')
          np.random.shuffle(imglist_val)
          current_index = 0

        for index in range(batch_size):
          imagepath, label = imglist_val[current_index+index].split()

          #input
          img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
          img_resize = cv2.resize(img, (input_width, input_height))
          img_data = (img_resize.astype(np.float32) - mean) * scale
          X.append(img_data)

          #label
          output_data = np.zeros(1000)
          output_data[int(label)] = 1.0
          Y.append(output_data)

        current_index += batch_size

        X = np.array(X)
        Y = np.array(Y)
        yield { 'inputs': X.reshape((batch_size,input_height,input_width,input_channel)), 'labels': Y.reshape((batch_size, 1000)) }      


def get_val_acc(batch_size, accuracy, x, y, is_training):
  generator_val = data_generator_val(batch_size)
  iter_num = int(img_num_val/batch_size)
  all_test_acc_val = []
  test_acc = 0
  for i in range(iter_num):
    data_batch_val = next(generator_val)
    inputs_val = data_batch_val['inputs']
    labels_val = data_batch_val['labels']

    train_accuracy = accuracy.eval(feed_dict={x: inputs_val, y: labels_val, is_training: False})
    all_test_acc_val.append(train_accuracy)
    test_acc = np.mean(all_test_acc_val)

    print('iter: %d/%d, batch_size: %d, current_batch accuracy: %g, training accuracy %g' % (i, iter_num, batch_size, train_accuracy, test_acc))
  return test_acc


#--------------------主函数---------------------#
def main(_):
  #create model
  model = create_model()
  x = model.x
  y = model.y
  is_training = model.is_training
  cross_entropy = model.cross_entropy
  logits = model.logits
  accuracy = model.accuracy
  dict_weightname_val = model.dict_weightname_val

  #create solver
  solver = create_admm_solver(model)
  dict_weightname_pattern = solver.dict_weightname_pattern
  dict_weightname_connect = solver.dict_weightname_connect
  train_step = solver.global_step
  ops_admm = solver.ops_admm

  #create patterns
  genPatterns = get_patterns(8)

  #data generator
  generator = data_generator(128)

  finetune = tf.train.AdamOptimizer(1e-4)
  grads = finetune.compute_gradients(cross_entropy)
  
  prun_rate = 50.0

  # 指定TFrecords路径，得到training iterator。
  # train_tfrecords = '../TRAIN_DATA/img_train.tfrecords'
  # train_iterator = data_iterator(train_tfrecords)
  # train_batch = train_iterator.get_next()

  model_path = '../../TRAIN_MODEL/' 

  saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=20)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    print("start restore weight!")
    ckpt = tf.train.latest_checkpoint(model_path)
    saver.restore(sess, ckpt)  #0.435397

    #reset train_step
    sess.run(train_step.assign(0))

    #get patterns
    ijj = 0
    for weight_name, val in dict_weightname_val.items():
      weight_val = sess.run(val)
      genPatterns.init_patterns(weight_val)
      print("get patterns! %d/%d" % (ijj, len(dict_weightname_val)))
      ijj += 1
    genPatterns.gen_patterns()

    #pattern regular terms
    dict_Z = {}
    dict_U = {}
    dict_pattern_mask = {}

    #connetivity pruned terms
    dict_Y = {}
    dict_V = {}
    dict_connect_mask = {}

    feed_dict = {}
    print("start select best pattern and pruning connectivity!")
    ijj = 0
    for weight_name, val in dict_weightname_val.items():
      print("layer: %d/%d" % (ijj, len(dict_weightname_val)), end='\r')
      ijj += 1

      weight_val = sess.run(val)

      #pattern select
      index, _, Z = genPatterns.get_best_pattern(weight_val)
      U = np.zeros_like(Z)
      dict_Z[weight_name] = Z
      dict_U[weight_name] = U
      feed_dict[dict_weightname_pattern[weight_name][0]] = Z
      feed_dict[dict_weightname_pattern[weight_name][1]] = U

      #connectivity pruned
      index, _, Y = pruning_bad_connect(weight_val, prun_rate)
      V = np.zeros_like(Y)
      dict_Y[weight_name] = Y
      dict_V[weight_name] = V
      feed_dict[dict_weightname_connect[weight_name][0]] = Y
      feed_dict[dict_weightname_connect[weight_name][1]] = V

    #while True:
    j = 0
    while j<4:
      j += 1
      for i in range(10000):
        #inputs, labels = sess.run(train_batch)
        data_batch = next(generator)
        inputs = data_batch['inputs']
        labels = data_batch['labels']
        feed_dict[x] = inputs
        feed_dict[y] = labels
        feed_dict[is_training] = True
        _, cross_entropy, zeroRegular_loss, pru_loss, learning_rate, step = sess.run(ops_admm, feed_dict)

        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: inputs, y: labels, is_training: False})
          print('epoch %d, step %d, cross_entropy %f, zeroRegular_loss %f, pru_loss %f, learning_rate %f, training_accuracy %g' % 
                                                                    (i/10000, step, cross_entropy, zeroRegular_loss, pru_loss, learning_rate, train_accuracy))
      
        if (i+1)%5000==0:
          saver.save(sess,'../../PRUNED_MODEL/' + 'resnet50_Sparse_Model_epoch_'+ str(j) + '_' + str(int((i+1)/5000)) + '.ckpt')
          # print(saver._var_list)
          val_acc = get_val_acc(128, accuracy, x, y, is_training)
          print("val_acc: %g" % (val_acc))
          
          # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['logits'])
          # # 写入序列化的 PB 文件
          # with tf.gfile.FastGFile('../../PRUNED_MODEL/' + 'resnet50_pretrainModel_epoch_'+ str(int(i/5000)) +'.pb', mode='wb') as f:
          #   f.write(constant_graph.SerializeToString())

      ijj = 0
      print("start select best pattern and pruning connectivity!")
      for weight_name, val in dict_weightname_val.items():
        print("layer: %d/%d" % (ijj, len(dict_weightname_val)), end='\r')
        ijj += 1

        weight_val = sess.run(val)

        #pattern select
        Z = weight_val + dict_U[weight_name]
        index, pattern_mask, Z = genPatterns.get_best_pattern(Z)
        U = dict_U[weight_name] + weight_val - Z
        dict_Z[weight_name] = Z
        dict_U[weight_name] = U
        feed_dict[dict_weightname_pattern[weight_name][0]] = Z
        feed_dict[dict_weightname_pattern[weight_name][1]] = U

        #connectivity pruned
        Y = weight_val + dict_V[weight_name]
        index, connect_mask, Y = pruning_bad_connect(Y, prun_rate)
        V = dict_V[weight_name] + weight_val - Y
        dict_Y[weight_name] = Y
        dict_V[weight_name] = V
        feed_dict[dict_weightname_connect[weight_name][0]] = Y
        feed_dict[dict_weightname_connect[weight_name][1]] = V

        dict_pattern_mask[weight_name] = pattern_mask
        dict_connect_mask[weight_name] = connect_mask

    dict_nzidx = apply_prune(dict_weightname_val, dict_pattern_mask, dict_connect_mask, sess)
    grads = apply_prune_on_grads(grads, dict_nzidx)
    apply_gradient_op = finetune.apply_gradients(grads)

    print ("init variables!")
    for i in range(len(tf.global_variables())): #645start
      print("global varials init %d/%d" % (i, len(tf.global_variables())))
      var = (tf.global_variables())[i]
      if tf.is_variable_initialized(var).eval() == False:
          print(var)
          sess.run(tf.variables_initializer([var]))

    print ("start retraining after pruning")
    ops_pru = [apply_gradient_op, cross_entropy]
    for i in range(10000*6):
        #inputs, labels = sess.run(train_batch)
        data_batch = next(generator)
        inputs = data_batch['inputs']
        labels = data_batch['labels']
        min_op, train_loss = sess.run(ops_pru, feed_dict = {x: inputs, y: labels, is_training: True})

        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={x: inputs, y: labels, is_training: False})
          print('epoch %d, step %d, train_loss %g, training accuracy %g' % (i/10000, i, train_loss, train_accuracy))
      
        if (i+1)%5000==0:
          saver.save(sess,'../../PRUNED_MODEL/' + 'resnet50_Pruning_Model_epoch_'+ str(int((i+1)/5000)) + '.ckpt')
          # print(saver._var_list)

          val_acc = get_val_acc(128, accuracy, x, y, is_training)
          print("val_acc: %g" % (val_acc))
          
          # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['logits'])
          # # 写入序列化的 PB 文件
          # with tf.gfile.FastGFile('../../PRUNED_MODEL/' + 'resnet50_pretrainModel_epoch_'+ str(int(i/5000)) +'.pb', mode='wb') as f:
          #   f.write(constant_graph.SerializeToString())

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['logits'])
    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile('../../PRUNED_MODEL/' + 'resnet50_Pruning_Model.pb', mode='wb') as f:
      f.write(constant_graph.SerializeToString())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  