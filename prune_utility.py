import numpy as np
import tensorflow as tf

def apply_prune(dict_weightname_val, dict_pattern_mask, dict_connect_mask, sess):
    # returns dictionary of non_zero_values' indices                                                                                                                   
  dict_nzidx = {}
  for target_name, weight in dict_weightname_val.items():
    print ("at weight "+target_name)
    weight_arr = sess.run(weight)
    print ("before pruning #non zero parameters " + str(np.sum(weight_arr!=0)))
    before = np.sum(weight_arr!=0)

    mask = dict_pattern_mask[target_name] * dict_connect_mask[target_name]
    weight_arr_pruned = mask * weight_arr

    after = np.sum(weight_arr_pruned!=0)
    print ("pruned "+ str(before-after))
    
    print ("after prunning #non zero parameters " + str(np.sum(weight_arr_pruned!=0)))
    sess.run(weight.assign(weight_arr_pruned))
    dict_nzidx[target_name] = mask
  return dict_nzidx


def apply_prune_on_grads(grads_and_vars, dict_nzidx):
  print("start apply prune on grads!")
  ijj = 0
  for key, nzidx in dict_nzidx.items():  
    print("%d/%d" % (ijj, len(dict_nzidx)))
    ijj += 1

    count = 0
    for grad, var in grads_and_vars:
      if var.name == key:
        print(var.name)
        nzidx_obj = tf.cast(tf.constant(nzidx), tf.float32)
        grads_and_vars[count] = (tf.multiply(nzidx_obj, grad), var)
      count += 1

  return grads_and_vars
  
