import tensorflow as tf
import numpy as np
import time

def main():
  start = time.time()
  train_x = np.linspace(-1,1,101)
  train_y = 36.0 + 2 * train_x + np.random.randn(*train_x.shape) * .33

  x = tf.placeholder("float")
  y = tf.placeholder("float")

  W = tf.Variable(0.0)
  b = tf.Variable(0.0)

  y_pred = tf.add(tf.mul(x, W), b)  

  cost = tf.reduce_mean(tf.pow(y - y_pred, 2)) # mean square error
  #cost = tf.reduce_mean(tf.abs(y-y_pred)) # mean absolute error
  #cost = tf.add( tf.reduce_mean(tf.pow(y-y_pred,2)), tf.mul(1.0,W)) # regularization
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)

  batch_start = time.time()
  for epoch in range(300):
    for (single_x,single_y) in zip(train_x,train_y):
      sess.run(train_step, feed_dict={x: single_x, y: single_y})

    if epoch % 100 == 0:
      mse = sess.run(cost,feed_dict={x: train_x, y: train_y})
      t = time.time() - batch_start
      batch_start = time.time()
      print "epoch: %s, error: %.3f, time/batch: %.3f" % (epoch, mse, t)

  
    # shuffle the data
    perm = np.arange(100)
    np.random.shuffle(perm)
    train_x = train_x[perm]
    train_y = train_y[perm]
      
  print sess.run(W)
  print sess.run(b)

if __name__ == "__main__":
  main()

