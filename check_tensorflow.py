import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Check if TensorFlow can perform a simple calculation
a = tf.constant(5)
b = tf.constant(3)

print("5 + 3 =", tf.add(a, b).numpy())