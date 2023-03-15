import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with tf.device('/device:GPU:0'):
  node1 = tf.constant(3.0, dtype=tf.float32)
  node2 = tf.constant(4.0, dtype=tf.float32)
  additionNode = tf.add(node1, node2)
  sess = tf.Session()
  result = sess.run(additionNode)
  print(result)
