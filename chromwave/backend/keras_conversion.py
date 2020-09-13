##############################
##############################
##############################
# This converts keras 2 models to deeplift 0.4.0 models.
# This required changes in the deeplift/backend/theano_backend in regards to module import of thano tensor signal pool funcitons.
##############################



# careful when trying to extend for dilated convolutions, there is a problem in the theano backend for border_mode='same'.
# shouldnt be a problem since we are unsing tensorflow but I'not sure.
# see https://github.com/fchollet/keras/issues/5090


##############################
##############################
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from deeplift.conversion.keras_conversion import *


from deeplift.blobs import SingleInputMixin, Node


from collections import OrderedDict, defaultdict
import deeplift
from deeplift import models, blobs
from deeplift.blobs import NonlinearMxtsMode, \
    DenseMxtsMode, ConvMxtsMode, MaxPoolDeepLiftMode
import deeplift.util

import numpy as np

from keras.utils.conv_utils import convert_kernel
import theano.tensor as T
# for some reason theano tensor does not import signal and then throws error, so have to manually import here
import theano.tensor.signal.pool
##############################
##############################
# Additional conversion for bespoke layers, globalpooling and
# clobal averaging and padding layer
#############################




def pad(x, width, val=0, batch_ndim=1):
    """
    Pad a tensor with a constant value.
    Parameters
    ----------
    x : tensor
    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.
    val : float
        The constant value used for padding
    batch_ndim : integer
        Dimensions before the value will not be padded.
    """
    input_shape = x.shape
    input_ndim = x.ndim

    output_shape = list(input_shape)
    indices = [slice(None) for _ in output_shape]

    if isinstance(width, int):
        widths = [width] * (input_ndim - batch_ndim)
    else:
        widths = width

    for k, w in enumerate(widths):
        try:
            l, r = w
        except TypeError:
            l = r = w
        output_shape[k + batch_ndim] += l + r
        indices[k + batch_ndim] = slice(l, l + input_shape[k + batch_ndim])

    if val:
        out = T.ones(output_shape) * val
    else:
        out = T.zeros(output_shape)
    return T.set_subtensor(out[tuple(indices)], x)



def disregard_paddding(x, width, batch_ndim):
    """
    Pad a tensor with a constant value.
    Parameters
    ----------
    x : tensor
    width : int, iterable of int, or iterable of tuple
        Padding width. If an int, pads each axis symmetrically with the same
        amount in the beginning and end. If an iterable of int, defines the
        symmetric padding width separately for each axis. If an iterable of
        tuples of two ints, defines a seperate padding width for each beginning
        and end of each axis.
    val : float
        The constant value used for padding
    batch_ndim : integer
        Dimensions before the value will not be padded.
    """
    input_shape = x.shape
    input_ndim = x.ndim

    output_shape = list(input_shape)
    indices = [slice(None) for _ in output_shape]

    if isinstance(width, int):
        widths = [width] * (input_ndim - batch_ndim)
    else:
        widths = width

    for k, w in enumerate(widths):

        try:
            l, r = w
        except TypeError:
            l = r = w
        output_shape[k + batch_ndim] -= (l + r)

        print(output_shape)
        indices[k + batch_ndim] = slice(l, input_shape[k + batch_ndim] - l)
        print(indices)


    return x[tuple(indices)]


class Pad1D(SingleInputMixin, Node):

    def __init__(self, padding=1, padding_value=0, batch_ndim =1, **kwargs):
        super(Pad1D, self).__init__(**kwargs)
        self.padding = padding
        self.padding_value = padding_value
        self.batch_ndim = batch_ndim

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Pad1D, self). \
            get_yaml_compatible_object_kwargs()
        kwargs_dict['padding'] = self.padding
        return kwargs_dict

    def _compute_shape(self, input_shape):
        # shape_to_return = [None, input_shape[0]] #channel axis the same
        # for dim_inp_len, dim_pad in zip(input_shape[2:], self.padding):
        #     shape_to_return.append(dim_inp_len + 2*dim_pad)
        # return shape_to_return
        output_shape = list(input_shape)

        if not isinstance(self.padding, int):
            widths = [0] * (len(input_shape) - self.batch_ndim)
            widths[0]=self.padding[0]
        else:
            widths = [self.padding]

        for k, w in enumerate(widths):
            if output_shape[k + self.batch_ndim] is None:
                continue
            else:
                try:
                    l, r = w
                except TypeError:
                    l = r = w
                output_shape[k + self.batch_ndim] += l + r
        return tuple(output_shape)

    def _build_activation_vars(self, input_act_vars):
        #Dimensions before the value will not be padded.
        return pad(input_act_vars, width=(self.padding,0), val=self.padding_value, batch_ndim=self.batch_ndim)

    def _get_mxts_increments_for_inputs(self):
        pos_mxts_increments=disregard_paddding(self.get_pos_mxts(), width=(self.padding,0), batch_ndim=self.batch_ndim)
        neg_mxts_increments=disregard_paddding(self.get_neg_mxts(), width=(self.padding,0), batch_ndim=self.batch_ndim)
        return pos_mxts_increments,neg_mxts_increments

    def _build_pos_and_neg_contribs(self):
        inp_pos_contribs, inp_neg_contribs =self._get_input_pos_and_neg_contribs()
        pos_contribs = self._build_activation_vars(inp_pos_contribs)
        neg_contribs = self._build_activation_vars(inp_neg_contribs)
        return pos_contribs, neg_contribs


class Pad2D(SingleInputMixin, Node):

    def __init__(self, padding, padding_value, batch_ndim, **kwargs):
        super(Pad2D, self).__init__(**kwargs)
        self.padding = padding
        self.padding_value = padding_value
        self.batch_ndim = batch_ndim

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Pad2D, self). \
            get_yaml_compatible_object_kwargs()
        kwargs_dict['padding'] = self.padding
        return kwargs_dict

    def _compute_shape(self, input_shape):
        # shape_to_return = [None, input_shape[0]] #channel axis the same
        # for dim_inp_len, dim_pad in zip(input_shape[2:], self.padding):
        #     shape_to_return.append(dim_inp_len + 2*dim_pad)
        # return shape_to_return
        output_shape = list(input_shape)

        if isinstance(self.padding, int):
            widths = [self.padding] * (len(input_shape) - self.batch_ndim)
        else:
            widths = self.padding

        for k, w in enumerate(widths):
            if output_shape[k + self.batch_ndim] is None:
                continue
            else:
                try:
                    l, r = w
                except TypeError:
                    l = r = w
                output_shape[k + self.batch_ndim] += l + r
        return tuple(output_shape)

    def _build_activation_vars(self, input_act_vars):
        return pad(input_act_vars, width=self.padding, val=self.padding_value, batch_ndim=self.batch_ndim)

    def _get_mxts_increments_for_inputs(self):
        return disregard_paddding(self.get_mxts(), width=self.padding, batch_ndim=self.batch_ndim)





def pad1d_conversion(layer, name, verbose,**kwargs):

    return [Pad1D(
        name=name,
        verbose=verbose,
        padding= layer.padding[0],
        padding_value=layer.value,
        batch_ndim = 1)]


def prep_globalpool1d_kwargs(layer, name, verbose):
    if (KerasKeys.border_mode in layer.get_config()):
        border_mode = layer.get_config()[KerasKeys.padding]
    else:
        border_mode = 'valid'

    if KerasKeys.strides in layer.get_config():
        strides = layer.get_config()[KerasKeys.strides][0]
    elif KerasKeys.stride in layer.get_config():
        strides = layer.get_config()[KerasKeys.stride][0]
    else:
        strides=1
    return {'name': name,
            'verbose': verbose,
            'pool_length':layer.input_shape[1],
            'stride': strides,
            'border_mode': border_mode,
            'ignore_border': False,
            'channels_come_last': True  # always applies to the 1D ops
            }


def globalmaxpooling1d_conversion(layer, name, verbose,
                         maxpool_deeplift_mode, **kwargs):
    pool1d_kwargs = prep_globalpool1d_kwargs(layer=layer, name=name, verbose=verbose)
    return [blobs.MaxPool1D(
        maxpool_deeplift_mode=maxpool_deeplift_mode,
        **pool1d_kwargs)]


def globalavgpooling1d_conversion(layer, name, verbose, **kwargs):
    pool1d_kwargs = prep_pool1d_kwargs(layer=layer, name=name, verbose=verbose)
    return [blobs.AvgPool1D(**pool1d_kwargs)]


KerasKeys = deeplift.util.enum(name='name', activation='activation',
                               subsample='subsample', subsample_length='subsample_length',
                               border_mode='border_mode', output_dim='output_dim',
                               pool_length='pool_length', stride='stride',
                               pool_size='pool_size', strides='strides',
                               padding='padding',
                               dim_ordering='dim_ordering',
                               mode='mode', concat_axis='concat_axis')

ActivationTypes = deeplift.util.enum(relu='relu',
                                     prelu='prelu',
                                     sigmoid='sigmoid',
                                     softmax='softmax',
                                     linear='linear')

default_maxpool_deeplift_mode = MaxPoolDeepLiftMode.gradient


def gru_conversion(layer, name, verbose, **kwargs):
    return [blobs.GRU(
        name=name,
        verbose=verbose,
        hidden_states_exposed=layer.get_config()['return_sequences'],
        weights_lookup=OrderedDict([
            ('weights_on_x_for_z', layer.W_z.copy()),
            ('weights_on_x_for_r', layer.W_r.copy()),
            ('weights_on_x_for_h', layer.W_h.copy()),
            ('weights_on_h_for_z', layer.U_z.copy()),
            ('weights_on_h_for_r', layer.U_r.copy()),
            ('weights_on_h_for_h', layer.U_h.copy()),
            ('bias_for_z', layer.b_z.copy()),
            ('bias_for_r', layer.b_r.copy()),
            ('bias_for_h', layer.b_h.copy()),
        ]),
        gate_activation_name=layer.get_config()['inner_activation'],
        hidden_state_activation_name=layer.get_config()['activation'])
    ]


def batchnorm_conversion(layer, name, verbose, **kwargs):
    import keras
    if (hasattr(keras, '__version__')):
        keras_version = float(keras.__version__[0:3])
    else:
        keras_version = 0.2
    if (keras_version <= 0.3):
        std = np.array(layer.running_std.get_value())
        epsilon = layer.epsilon
    else:
        std = np.sqrt(np.array(layer.running_std.get_value() + layer.epsilon))
        epsilon = 0
    return [blobs.BatchNormalization(
        name=name,
        verbose=verbose,
        gamma=np.array(layer.gamma.get_value()),
        beta=np.array(layer.beta.get_value()),
        axis=layer.axis,
        mean=np.array(layer.running_mean.get_value()),
        std=std,
        epsilon=epsilon)]

import numpy
def conv1d_conversion(layer, name, verbose,
                      nonlinear_mxts_mode, conv_mxts_mode, **kwargs):
    # nonlinear_mxts_mode only used for activation
    converted_activation = activation_conversion(
        layer, name, verbose,
        nonlinear_mxts_mode=nonlinear_mxts_mode)
    #
    W = convert_kernel(layer.get_weights()[0].astype(numpy.float64))


    # ahouls bw 16,4,24

    # w is expected to have shape  num_output_channels x num_inp_channels x cols_kern_width
    # N,channel , height, width
    # # if (W.shape[-1] != 1):  # is NHWC and not NCHW - need to transpose
    # W =W.transpose(2,1,0)

    # if (W.shape[-1] != 1): #is NHWC and not NCHW - need to transpose
    #W = W.transpose(3,2,0,1)
    to_return = [blobs.Conv1D(
        name=("preact_" if len(converted_activation) > 0
              else "") + name,
        #W=np.squeeze(W, 1),
        # #W=layer.get_weights()[0],
        #W=np.squeeze(W, -1),
        W=W.transpose(2, 1, 0),
        b=layer.get_weights()[1].astype(numpy.float64),
        stride=layer.get_config()[KerasKeys.strides][0],
        border_mode=layer.get_config()[KerasKeys.padding],
        # for conv1d implementations, channels always seem to come last
        channels_come_last=True,
        conv_mxts_mode=conv_mxts_mode
    )]
    to_return.extend(converted_activation)
    return to_return



def conv2d_conversion(layer, name, verbose,
                      nonlinear_mxts_mode, conv_mxts_mode, **kwargs):
    # nonlinear_mxts_mode only used for activation
    converted_activation = activation_conversion(
        layer, name, verbose,
        nonlinear_mxts_mode=nonlinear_mxts_mode)

    W = convert_kernel(layer.get_weights()[0].astype(numpy.float64))
    channels_come_last = False
    if KerasKeys.dim_ordering in layer.get_config():
        dim_ordering = layer.get_config()[KerasKeys.dim_ordering]
        if (dim_ordering == 'tf'):
            W = W.transpose(3, 2, 0, 1)
            channels_come_last = True
    to_return = [blobs.Conv2D(
        name=("preact_" if len(converted_activation) > 0
              else "") + name,
        W=W,
        b=layer.get_weights()[1],
        strides=layer.get_config()[KerasKeys.strides],
        border_mode=layer.get_config()[KerasKeys.padding],
        channels_come_last=channels_come_last,
        conv_mxts_mode=conv_mxts_mode)]
    to_return.extend(converted_activation)
    return to_return


def prep_pool2d_kwargs(layer, name, verbose):
    channels_come_last = False
    if (KerasKeys.dim_ordering in layer.get_config()):
        dim_ordering = layer.get_config()[KerasKeys.dim_ordering]
        if (dim_ordering in 'tf'):
            channels_come_last = True
    if KerasKeys.strides in layer.get_config():
        strides = layer.get_config()[KerasKeys.strides]
    elif KerasKeys.stride in layer.get_config():
        strides = layer.get_config()[KerasKeys.stride]
    else:
        raise RuntimeError("Unsure how to get strides argument")

    if (KerasKeys.border_mode in layer.get_config()):
        border_mode = layer.get_config()[KerasKeys.padding]
    else:
        border_mode = 'valid'

    return {'name': name,
            'verbose': verbose,
            'pool_size': layer.get_config()[KerasKeys.pool_size],
            'strides': strides,
            'border_mode': border_mode,
            'ignore_border': True,
            'channels_come_last': channels_come_last}





def prep_pool1d_kwargs(layer, name, verbose):
    if (KerasKeys.padding in layer.get_config()):
        border_mode = layer.get_config()[KerasKeys.padding]
    else:
        border_mode = 'valid'

    if KerasKeys.strides in layer.get_config():
        strides = layer.get_config()[KerasKeys.strides][0]
    elif KerasKeys.stride in layer.get_config():
        strides = layer.get_config()[KerasKeys.stride][0]
    return {'name': name,
            'verbose': verbose,
            'pool_length': layer.get_config()[KerasKeys.pool_size][0],
            'stride': strides,
            'border_mode': border_mode,
            'ignore_border': True,
            'channels_come_last': True  # always applies to the 1D ops
            }


def maxpool1d_conversion(layer, name, verbose,
                         maxpool_deeplift_mode, **kwargs):
    pool1d_kwargs = prep_pool1d_kwargs(layer=layer, name=name, verbose=verbose)
    return [blobs.MaxPool1D(
        maxpool_deeplift_mode=maxpool_deeplift_mode,
        **pool1d_kwargs)]


def avgpool1d_conversion(layer, name, verbose, **kwargs):
    pool1d_kwargs = prep_pool1d_kwargs(layer=layer, name=name, verbose=verbose)
    return [blobs.AvgPool1D(**pool1d_kwargs)]


def dropout_conversion(name, **kwargs):
    return [blobs.NoOp(name=name)]


def flatten_conversion(layer, name, verbose, **kwargs):
    return [blobs.Flatten(name=name, verbose=verbose)]


def dense_conversion(layer, name, verbose,
                     dense_mxts_mode, nonlinear_mxts_mode, **kwargs):
    converted_activation = activation_conversion(
        layer, name=name, verbose=verbose,
        nonlinear_mxts_mode=nonlinear_mxts_mode)
    to_return = [blobs.Dense(
        name=("preact_" if len(converted_activation) > 0
              else "") + name,
        verbose=verbose,
        W=layer.get_weights()[0],
        b=layer.get_weights()[1],
        dense_mxts_mode=dense_mxts_mode)]
    to_return.extend(converted_activation)
    return to_return


def linear_conversion(**kwargs):
    return []


def prelu_conversion(layer, name, verbose, nonlinear_mxts_mode, **kwargs):
    return [blobs.PReLU(alpha=layer.get_weights()[0],
                        name=name, verbose=verbose,
                        nonlinear_mxts_mode=nonlinear_mxts_mode)]


def relu_conversion(layer, name, verbose, nonlinear_mxts_mode):
    return [blobs.ReLU(name=name, verbose=verbose,
                       nonlinear_mxts_mode=nonlinear_mxts_mode)]


def sigmoid_conversion(layer, name, verbose, nonlinear_mxts_mode):
    return [blobs.Sigmoid(name=name, verbose=verbose,
                          nonlinear_mxts_mode=nonlinear_mxts_mode)]


def softmax_conversion(layer, name, verbose,
                       nonlinear_mxts_mode):
    return [blobs.Softmax(name=name, verbose=verbose,
                          nonlinear_mxts_mode=nonlinear_mxts_mode)]


def activation_conversion(layer, name, verbose, nonlinear_mxts_mode, **kwargs):
    activation = layer.get_config()[KerasKeys.activation]

    return activation_to_conversion_function(activation)(
        layer=layer, name=name, verbose=verbose,
        nonlinear_mxts_mode=nonlinear_mxts_mode)


def merge_conversion(layer, name, verbose, **kwargs):
    if layer.get_config()[KerasKeys.mode] == "concat":
        return [blobs.core.Concat(
            axis=layer.get_config()[KerasKeys.concat_axis],
            name=name, verbose=verbose)]
    else:
        raise RuntimeError("Unsupported merge mode: "
                           + str(layer.get_config()[KerasKeys.mode]))


def sequential_container_conversion(layer, name, verbose,
                                    nonlinear_mxts_mode,
                                    dense_mxts_mode,
                                    conv_mxts_mode,
                                    maxpool_deeplift_mode,
                                    converted_layers=None,
                                    layer_overrides={}):
    if (converted_layers is None):
        converted_layers = []
    # rename some arguments to be more intuitive
    container = layer
    name_prefix = name
    for layer_idx, layer in enumerate(container.layers):
        layer_type=type(layer).__name__

        if not layer_type == 'InputLayer':

            if layer_type == 'Dense':
                # if more than one leading axis than add a FlattenLayer first before adding the dense layer
                if len(layer.input_shape) > 2 or converted_layers[-1].__class__.__name__ in ['AvgPool2D', 'MaxPool2D', 'AvgPool1D', 'MaxPool1D']:
                    print('Inserting Flatten Layer in front of ' + type(layer).__name__)
                    conversion_function = layer_name_to_conversion_function('flatten')
                    converted_layers.extend(conversion_function(
                        layer=layer,
                        name=(name_prefix + "-" if name_prefix != ""
                              else "") + 'Flatten_' + str(layer_idx),
                        verbose=verbose,
                        dense_mxts_mode=dense_mxts_mode,
                        nonlinear_mxts_mode=nonlinear_mxts_mode,
                        maxpool_deeplift_mode=maxpool_deeplift_mode))  # maxpool_deeplift+


            conversion_function = layer_name_to_conversion_function(
                type(layer).__name__)
            modes_to_pass = {'dense_mxts_mode': dense_mxts_mode,
                             'conv_mxts_mode': conv_mxts_mode,
                             'nonlinear_mxts_mode': nonlinear_mxts_mode,
                             'maxpool_deeplift_mode': maxpool_deeplift_mode
                             }
            if layer_idx in layer_overrides:
                for mode in ['dense_mxts_mode', 'conv_mxts_mode',
                             'nonlinear_mxts_mode']:
                    if mode in layer_overrides[layer_idx]:
                        modes_to_pass[mode] = layer_overrides[layer_idx][mode]

            converted_layers.extend(conversion_function(
                layer=layer,
                name=(name_prefix + "-" if name_prefix != ""
                      else "") + str(layer_idx),
                verbose=verbose,
                **modes_to_pass))
    return converted_layers


def activation_to_conversion_function(activation):
    activation_dict = {
        ActivationTypes.linear: linear_conversion,
        ActivationTypes.relu: relu_conversion,
        ActivationTypes.sigmoid: sigmoid_conversion,
        ActivationTypes.softmax: softmax_conversion
    }

    return activation_dict[activation]


def layer_name_to_conversion_function(layer_name):
    name_dict = {
        'inputlayer': input_layer_conversion,
        'conv1d': conv1d_conversion,
        'maxpooling1d': maxpool1d_conversion,
        'globalmaxpooling1d': globalmaxpooling1d_conversion,
        'averagepooling1d': avgpool1d_conversion,
        'globalaveragepooling1d': globalavgpooling1d_conversion,
        'conv2d': conv2d_conversion,
        'maxpooling2d': maxpool2d_conversion,
        'averagepooling2d': avgpool2d_conversion,
        'batchnormalization': batchnorm_conversion,
        'flatten': flatten_conversion,
        'dense': dense_conversion,
        'pad1d': pad1d_conversion,
        # in current keras implementation, scaling is done during training
        # and not predict time, so dropout is a no-op at predict time
        'dropout': dropout_conversion,
        'activation': activation_conversion,
        'prelu': prelu_conversion,
        'sequential': sequential_container_conversion,
        'merge': merge_conversion
    }

    # lowercase to create resistance to capitalization changes
    # was a problem with previous Keras versions
    return name_dict[layer_name.lower()]


def convert_sequential_model(model,
                             num_dims=None,
                             nonlinear_mxts_mode= \
                                     NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
                             verbose=True,
                             dense_mxts_mode=DenseMxtsMode.Linear,
                             conv_mxts_mode=ConvMxtsMode.Linear,
                             maxpool_deeplift_mode=default_maxpool_deeplift_mode,
                             layer_overrides={}):
    if (verbose):
        print("nonlinear_mxts_mode is set to: " + str(nonlinear_mxts_mode))
    converted_layers = []
    if (model.layers[0].input_shape is not None):
        input_shape = model.layers[0].input_shape
        assert input_shape[0] is None  # batch axis
        num_dims_input = len(input_shape)
        assert num_dims is None or num_dims_input == num_dims, \
            "num_dims argument of " + str(num_dims) + " is incompatible with" \
            + " the number of dims in layers[0].input_shape which is: " \
            + str(model.layers[0].input_shape)
        num_dims = num_dims_input
    else:
        input_shape = None
    # if not (type(model.layers[0]).__name__ == "InputLayer"):
    converted_layers.append(
        blobs.Input(num_dims=num_dims, shape=input_shape, name="input"))
    # else:
    #     converted_layers.extend(input_layer_conversion(
    #         keras_input_layer=model.layers[0], layer_name="input"
    #         ))

    # converted_layers is actually mutated to be extended with the
    # additional layers so the assignment is not strictly necessary,
    # but whatever

    converted_layers = sequential_container_conversion(
        layer=model, name="", verbose=verbose,
        nonlinear_mxts_mode=nonlinear_mxts_mode,
        dense_mxts_mode=dense_mxts_mode,
        conv_mxts_mode=conv_mxts_mode,
        maxpool_deeplift_mode=maxpool_deeplift_mode,
        converted_layers=converted_layers,
        layer_overrides=layer_overrides)

    deeplift.util.connect_list_of_layers(converted_layers)
    converted_layers[-1].build_fwd_pass_vars()

    return models.SequentialModel(converted_layers)


def input_layer_conversion(keras_input_layer, layer_name):
    input_shape = keras_input_layer.get_config()['batch_input_shape']
    if (input_shape[0] is not None):
        input_shape = [None] + [x for x in input_shape]
    assert input_shape[0] is None  # for the batch axis
    deeplift_input_layer = \
        blobs.Input(shape=input_shape, num_dims=None,
                    name=layer_name)
    return deeplift_input_layer


def convert_graph_model(model,
                        nonlinear_mxts_mode= \
                                NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
                        verbose=True,
                        dense_mxts_mode=DenseMxtsMode.Linear,
                        conv_mxts_mode=ConvMxtsMode.Linear,
                        maxpool_deeplift_mode=default_maxpool_deeplift_mode,
                        auto_build_outputs=True):
    if (verbose):
        print("nonlinear_mxts_mode is set to: " + str(nonlinear_mxts_mode))
    name_to_blob = OrderedDict()
    keras_layer_to_deeplift_blobs = OrderedDict()
    keras_non_input_layers = []

    # convert the inputs
    for keras_input_layer_name in model.inputs:
        keras_input_layer = model.inputs[keras_input_layer_name]
        deeplift_input_layer = input_layer_conversion(
            keras_input_layer=keras_input_layer,
            layer_name=keras_input_layer_name)
        name_to_blob[keras_input_layer_name] = deeplift_input_layer
        keras_layer_to_deeplift_blobs[id(keras_input_layer)] = \
            [deeplift_input_layer]

    # convert the nodes/outputs
    for layer_name, layer in list(model.nodes.items()):
        # need some special handling when previous layer
        # is Merge as merge is not given its own node
        if (type(get_previous_layer(layer)).__name__ == 'Merge'):
            merge_layer = get_previous_layer(layer)
            keras_non_input_layers.append(merge_layer)
            deeplift_merge_layer = merge_conversion(
                layer=merge_layer,
                name='merge_before_' + layer_name,
                verbose=verbose)
            keras_layer_to_deeplift_blobs[id(merge_layer)] = \
                deeplift_merge_layer
            assert len(deeplift_merge_layer) == 1
            name_to_blob[deeplift_merge_layer[0].get_name()] = \
                deeplift_merge_layer[0]
        # now for converting the actual layer
        conversion_function = layer_name_to_conversion_function(
            type(layer).__name__)
        keras_non_input_layers.append(layer)
        deeplift_layers = conversion_function(
            layer=layer, name=layer_name,
            verbose=verbose,
            nonlinear_mxts_mode=nonlinear_mxts_mode,
            dense_mxts_mode=dense_mxts_mode,
            conv_mxts_mode=conv_mxts_mode,
            maxpool_deeplift_mode=maxpool_deeplift_mode)
        deeplift.util.connect_list_of_layers(deeplift_layers)
        keras_layer_to_deeplift_blobs[id(layer)] = deeplift_layers
        for deeplift_layer in deeplift_layers:
            name_to_blob[deeplift_layer.get_name()] = deeplift_layer

    # connect any remaining things not connected to their inputs
    for keras_non_input_layer in keras_non_input_layers:
        deeplift_layers = \
            keras_layer_to_deeplift_blobs[id(keras_non_input_layer)]
        previous_keras_layers = get_previous_layer(keras_non_input_layer)
        if (isinstance(previous_keras_layers, list)):
            previous_deeplift_layers = \
                [keras_layer_to_deeplift_blobs[id(x)][-1]
                 for x in previous_keras_layers]
            deeplift_layers[0].set_inputs(previous_deeplift_layers)
        else:
            previous_deeplift_layer = \
                keras_layer_to_deeplift_blobs[id(previous_keras_layers)][-1]
            deeplift_layers[0].set_inputs(previous_deeplift_layer)

    if (auto_build_outputs):
        for layer in model.outputs.values():
            layer_to_build = keras_layer_to_deeplift_blobs[id(layer)][-1]
            layer_to_build.build_fwd_pass_vars()
    return models.GraphModel(name_to_blob=name_to_blob,
                             input_layer_names=model.inputs.keys())


def get_previous_layer(keras_layer):
    if (hasattr(keras_layer, 'previous')):
        return keras_layer.previous
    elif (type(keras_layer).__name__ == 'Sequential'):
        return keras_layer.layers[0].previous
    elif (type(keras_layer).__name__ == 'Merge'):
        return keras_layer.layers
    else:
        raise RuntimeError("Not sure how to get prev layer for"
                           + " " + str(type(keras_layer)) + "; attributes: "
                           + str(dir(keras_layer)))


def convert_functional_model(
        model,
        nonlinear_mxts_mode= \
                NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
        verbose=True,
        dense_mxts_mode=DenseMxtsMode.Linear,
        conv_mxts_mode=ConvMxtsMode.Linear,
        maxpool_deeplift_mode=default_maxpool_deeplift_mode,
        auto_build_outputs=True):
    if (verbose):
        print("nonlinear_mxts_mode is set to: " + str(nonlinear_mxts_mode))
    keras_layer_name_to_keras_nodes = OrderedDict()
    for keras_node_depth in sorted(model.nodes_by_depth.keys()):
        keras_nodes_at_depth = model.nodes_by_depth[keras_node_depth]
        for keras_node in keras_nodes_at_depth:
            keras_layer_name = keras_node.outbound_layer.name
            if (keras_layer_name not in keras_layer_name_to_keras_nodes):
                keras_layer_name_to_keras_nodes[keras_layer_name] = []
            keras_layer_name_to_keras_nodes[keras_layer_name] \
                .append(keras_node)

    name_to_blob = OrderedDict()
    # for each node, we only know the input tensor ids. Thus, to figure out
    # which nodes link to which ids, we need to create a mapping from output
    # tensor ids to node ids (this is a many-to-one mapping as a single node
    # can have multiple output tensors)
    keras_tensor_id_to_node_id = {}
    keras_node_id_to_deeplift_blobs = {}
    for keras_layer_name in keras_layer_name_to_keras_nodes:
        keras_nodes = keras_layer_name_to_keras_nodes[keras_layer_name]
        num_nodes_for_layer = len(keras_nodes)
        for i, keras_node in enumerate(keras_nodes):
            # record the mapping from output tensor ids to node
            for keras_tensor in keras_node.output_tensors:
                keras_tensor_id_to_node_id[id(keras_tensor)] = id(keras_node)
            # figure out what to call the deeplift layer based on whether
            # there are multiple shared instances of the associated layer
            if (num_nodes_for_layer == 1):
                deeplift_layer_name = keras_layer_name
            else:
                deeplift_layer_name = keras_layer_name + "_" + str(i)

            keras_layer = keras_node.outbound_layer
            # convert the node
            if (type(keras_layer).__name__ == "InputLayer"):
                # there should be only one edition of each input layer
                # so the deeplift layer name and keras layer name should
                # converge
                assert deeplift_layer_name == keras_layer.name
                converted_deeplift_blobs = [input_layer_conversion(
                    keras_input_layer=keras_layer,
                    layer_name=deeplift_layer_name)]
            else:
                conversion_function = layer_name_to_conversion_function(
                    type(keras_layer).__name__)
                converted_deeplift_blobs = conversion_function(
                    layer=keras_layer,
                    name=deeplift_layer_name,
                    verbose=verbose,
                    nonlinear_mxts_mode=nonlinear_mxts_mode,
                    dense_mxts_mode=dense_mxts_mode,
                    conv_mxts_mode=conv_mxts_mode,
                    maxpool_deeplift_mode=maxpool_deeplift_mode)
                deeplift.util.connect_list_of_layers(converted_deeplift_blobs)
            # record the converted blobs in name_to_blob
            for blob in converted_deeplift_blobs:
                name_to_blob[blob.get_name()] = blob
                # record the converted blobs in the node id -> blobs dict
            keras_node_id_to_deeplift_blobs[id(keras_node)] = \
                converted_deeplift_blobs

    # link up all the blobs
    for node_depth in sorted(model.nodes_by_depth.keys()):
        for node in model.nodes_by_depth[node_depth]:
            if (type(node.outbound_layer).__name__ != "InputLayer"):
                input_node_ids = []
                # map the input tensors of each node to the node ids
                for input_tensor in node.input_tensors:
                    if id(input_tensor) not in input_node_ids:
                        input_node_ids.append(
                            keras_tensor_id_to_node_id[id(input_tensor)])
                # map the node ids to deeplift blobs
                # if a single node id translates into multiple deeplift blobs
                # (eg, if the node came with an activation), take the last
                # deeplift blob in the list.
                input_deeplift_blobs = [
                    keras_node_id_to_deeplift_blobs[x][-1] for x
                    in input_node_ids]
                # if there is only one input blob, unlist
                if (len(input_deeplift_blobs) == 1):
                    input_deeplift_blobs = input_deeplift_blobs[0]
                # link the inputs
                keras_node_id_to_deeplift_blobs[id(node)][0] \
                    .set_inputs(input_deeplift_blobs)

    if (auto_build_outputs):
        for output_node in model.nodes_by_depth[0]:
            layer_to_build = keras_node_id_to_deeplift_blobs[
                id(output_node)][-1]
            print(layer_to_build)
            layer_to_build.build_fwd_pass_vars()
    return models.GraphModel(name_to_blob=name_to_blob,
                             input_layer_names=model.input_names)


def mean_normalise_softmax_weights(softmax_dense_layer):
    weights, biases = softmax_dense_layer.get_weights()
    new_weights, new_biases = \
        deeplift.util.get_mean_normalised_softmax_weights(weights, biases)
    softmax_dense_layer.set_weights([new_weights, new_biases])


def load_keras_model(weights, yaml=None, json=None,
                     normalise_conv_for_one_hot_encoded_input=False,
                     axis_of_normalisation=None,
                     name_of_conv_layer_to_normalise=None):
    if (normalise_conv_for_one_hot_encoded_input):
        assert axis_of_normalisation is not None, \
            "specify axis of normalisation for normalising one-hot encoded input"
    assert yaml is not None or json is not None, \
        "either yaml or json must be specified"
    assert yaml is None or json is None, \
        "only one of yaml or json must be specified"
    if (yaml is not None):
        from keras.models import model_from_yaml
        model = model_from_yaml(open(yaml).read())
    else:
        from keras.models import model_from_json
        model = model_from_json(open(json).read())
    model.load_weights(weights)
    if (normalise_conv_for_one_hot_encoded_input):
        mean_normalise_first_conv_layer_weights(
            model,
            axis_of_normalisation=axis_of_normalisation,
            name_of_conv_layer_to_normalise=name_of_conv_layer_to_normalise)
    return model


def mean_normalise_first_conv_layer_weights(model,
                                            axis_of_normalisation,
                                            name_of_conv_layer_to_normalise):
    if (type(model).__name__ == "Sequential"):
        layer_to_adjust = model.layers[0];
    elif (type(model).__name__ == "Graph"):
        assert name_of_conv_layer_to_normalise is not None, \
            "Please provide name of conv layer for graph model"
        assert name_of_conv_layer_to_normalise in model.nodes, \
            name_of_conv_layer_to_normalise + " not found; node names are: " \
                                              " " + str(model.nodes.keys())
        layer_to_adjust = model.nodes[name_of_conv_layer_to_normalise]
    weights, biases = layer_to_adjust.get_weights();
    normalised_weights, normalised_bias = \
        deeplift.util.mean_normalise_weights_for_sequence_convolution(
            weights, biases, axis_of_normalisation,
            dim_ordering=layer_to_adjust.get_config()['dim_ordering'])
    layer_to_adjust.set_weights([normalised_weights,
                                 normalised_bias])





















