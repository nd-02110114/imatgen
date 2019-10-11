import tensorflow as tf

def threshold(x, val=0.5):
    x = tf.clip_by_value(x,0.5,0.5001) - 0.5
    x = tf.minimum(x * 10000,1) 
    return x

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)
