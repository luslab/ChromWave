
from keras import backend as K
import keras

class Pad1D(keras.engine.Layer):
    """Padding layer for 1D input (e.g. temporal sequence).
    # Arguments
        padding: int, or tuple of int (length 2), or dictionary.
            - If int:
            How many zeros to add at the beginning and end of
            the padding dimension (axis 1).
            - If tuple of int (length 2):
            How many zeros to add at the beginning and at the end of
            the padding dimension (`(left_pad, right_pad)`).
    # Input shape
        3D tensor with shape `(batch, axis_to_pad, features)`
    # Output shape
        3D tensor with shape `(batch, padded_axis, features)`
    """

    def __init__(self, padding=1,value =0,  **kwargs):
        super(Pad1D, self).__init__(**kwargs)
        self.padding = keras.utils.conv_utils.normalize_tuple(padding, 2, 'padding')
        self.value = value

        self.input_spec = keras.engine.InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):

        if input_shape[1] is not None:
            length = input_shape[1] + self.padding[0] + self.padding[1]
        else:
            length = None
        return (input_shape[0],
                length,
                input_shape[2])

    def call(self, inputs):
        # padding only works with zeros. Little trick to make it pad with value
        return K.temporal_padding(inputs-self.value, padding=self.padding)+self.value

    def get_config(self):
        config = {'padding': self.padding, 'value': self.value}
        base_config = super(Pad1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


