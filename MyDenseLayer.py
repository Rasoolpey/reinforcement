import tensorflow as tf

class MyDenseLayer(tf.keras.layers.layer):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()

        #initialize weghts and bias
        self.w = self.add_weight([input_dim, output_dim])
        self.b = self.add_weight([1, output_dim])
