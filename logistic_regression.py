import tensorflow as tf
import numpy as np
import input_data

def main():
  data = input_data.read_data_sets() 

  num_features = data.train.features.shape[1]
  
  # Graph input
  x = tf.placeholder("float", [None,num_features])
  y = tf.placeholder("float", [None, 1])

  # Model
  W = tf.Variable(tf.zeros([num_features,1]))
  b = tf.Variable(tf.zeros([1]))

  y_pred = tf.nn.sigmoid(tf.matmul(x, W) + b)
  
  # Optimization
  cost_func = -tf.reduce_sum(y*tf.log(y_pred))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost_func)

  # Initialize
  init = tf.initialize_all_variables()

  # Learn
  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1000):
      batch_xs, batch_ys = data.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

      if epoch % 100 == 0:
        # print some outputs
        correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        train_acc = accuracy.eval({x: data.train.features, y: data.train.targets})
        valid_acc = accuracy.eval({x: data.validation.features, y: data.validation.targets})
        
        #print "Epoch: %s, Train Acc: %.3f, Valid Acc: %.3f" % (epoch, train_acc, valid_acc)

