import tensorflow as tf

hello = tf.constant("HELLO")
sess = tf.Session()
print(sess.run(hello))

