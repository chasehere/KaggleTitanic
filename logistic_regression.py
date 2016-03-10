import tensorflow as tf
import numpy as np
import pandas as pd
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
  
  # Helper functions
  correct_prediction = tf.equal(tf.round(y_pred), y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
  
  # Optimization
  cost_func = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y) )
  train_step = tf.train.MomentumOptimizer(0.003,.5).minimize(cost_func)
  
  # Initialize
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  # Learn
  for epoch in range(2501):
    batch_xs, batch_ys = data.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

    if epoch % 500 == 0:
      # print some outputs
      train_cost = sess.run(cost_func,feed_dict={x: data.train.features, y: data.train.targets}) / data.train.num_examples
      valid_cost = sess.run(cost_func,feed_dict={x: data.validation.features, y: data.validation.targets}) / data.validation.num_examples
      train_acc = sess.run( accuracy, feed_dict={x: data.train.features, y: data.train.targets})
      valid_acc = sess.run( accuracy, feed_dict={x: data.validation.features, y: data.validation.targets})
      print "Epoch: %s, Train/Valid Cost: %.3f/%.3f, Train/Valid Accuracy: %.3f/%.3f" % (epoch, train_cost, valid_cost, train_acc, valid_acc)

 
  #train_pred = sess.run(y_pred, feed_dict={x: data.train.features, y: data.train.targets})
  #valid_pred = sess.run(y_pred, feed_dict={x: data.validation.features, y: data.validation.targets})
  test_pred = sess.run(y_pred, feed_dict={x: data.test.features, y: data.test.targets})

  print "Weights: %s" % sess.run(W)
  print "Intercept: %s" % sess.run(b)

  sess.close()

  # Write data to file
  test_pred = np.round(test_pred)
  df = pd.DataFrame({'PassengerId' : data.test.ids[:,0], 'Survived' : test_pred[:,0]})
  df.to_csv('data/tensorflow_benchmark.csv', index=False) 

if __name__ == "__main__":
  main()

