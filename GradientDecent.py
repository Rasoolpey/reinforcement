import tensorflow as tf

weights = tf.Variable([tf.random.normal()])

while True:
    with tf.GradientTape() as g:
        loss = compute_loss(weights)
        grsdient =g.gradient(loss, weights)
    
    weights = weights- Ir * gradient
    