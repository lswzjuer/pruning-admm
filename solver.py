import tensorflow as tf
import numpy as np

class AdmmSolver():
  def __init__(self,model):   
    dict_weightname_val = model.dict_weightname_val
    cross_entropy = model.cross_entropy

    #weight l2 loss
    weight_l2loss_sum = 0
    for weightname, val in dict_weightname_val.items():
        weight_l2loss_sum += tf.nn.l2_loss(val)

    #admm loss
    weight_pattern_loss = 0
    self.dict_weightname_pattern = {}
    for weightname, val in dict_weightname_val.items():
        Z = tf.placeholder(tf.float32, shape = val.get_shape().as_list())
        U = tf.placeholder(tf.float32, shape = val.get_shape().as_list())
        admm_term = [Z, U]
        self.dict_weightname_pattern[weightname] = admm_term
        weight_pattern_loss += tf.nn.l2_loss(val - Z + U)

    weight_connect_loss = 0
    self.dict_weightname_connect = {}
    for weightname, val in dict_weightname_val.items():
        Y = tf.placeholder(tf.float32, shape = val.get_shape().as_list())
        V = tf.placeholder(tf.float32, shape = val.get_shape().as_list())
        admm_term = [Y, V]
        self.dict_weightname_connect[weightname] = admm_term
        weight_connect_loss += tf.nn.l2_loss(val - Y + V)

    #change learning rate
    starter_learning_rate = 1e-2
    steps_per_decay = 10000*4
    decay_factor = 0.1
    self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay( learning_rate = starter_learning_rate,
                                                global_step = self.global_step,
                                                decay_steps = steps_per_decay,
                                                decay_rate = decay_factor,
                                                staircase = True  #If `True` decay the learning rate at discrete intervals
                                                #staircase = False,change learning rate at every step
                                              )

    with tf.name_scope( 'train_op' ): 
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        optimizer=tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy, global_step = self.global_step)
        train_step_zeroRegular = optimizer.minimize(cross_entropy + 5e-9 * weight_l2loss_sum, global_step = self.global_step)
        train_step_admm = optimizer.minimize(cross_entropy + 5e-9 * weight_l2loss_sum + 1e-8 * weight_pattern_loss + 1e-8 * weight_connect_loss, global_step = self.global_step)
    
    zeroRegular_loss = 5e-9 * weight_l2loss_sum 
    pru_loss = 1e-8 * weight_pattern_loss + 1e-8 * weight_connect_loss

    self.ops = [train_step, cross_entropy, learning_rate]
    self.ops_zeroRegular = [train_step_zeroRegular, cross_entropy, learning_rate]
    self.ops_admm = [train_step_admm, cross_entropy, zeroRegular_loss, pru_loss, learning_rate, self.global_step]

def create_admm_solver(model):
  return AdmmSolver(model)

