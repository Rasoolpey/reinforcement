import tensorflow as tf
tf.keras.layers.Dense(units=2)
model=tf.keras.Sequential([tf.keras.layers.Dense(n1),tf.keras.layers.Dense(n2),tf.keras.layers.Dense(2)])
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,predicted))
loss=tf.reduce_mean(tf.square(tf.subtract(y,predicted)))
loss=tf.keras.losses.MSE(y,predicted)
