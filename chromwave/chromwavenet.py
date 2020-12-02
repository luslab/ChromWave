from copy import copy
from itertools import product
import inspect
import keras
import keras.backend as K
from keras import regularizers, layers
from keras.layers.merge import Multiply, Add, Average, Concatenate
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers import Adam as adam
from keras.optimizers import SGD as sgd
from keras.optimizers import Nadam as nadam
from keras.utils import plot_model as kplot_model
from keras import metrics as kmetrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from matplotlib import pyplot as plt
import numpy
import os
from operator import add
import pandas
import pickle
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import confusion_matrix
import subprocess
import tensorflow as tf
from tensorflow.python.keras.backend import _to_tensor

from .functions import utils, load_weights
from .functions.signal_processing import int_to_float
from .vis import motif_viz, plot,saliency
from .vis.keras_utils_vis_utils import plot_model as kplot_model
from .backend import keras_backend as kfb




_EPSILON = K.epsilon()

def load_existing_wavenet_conv_weights(directory):
    """
    Function to load the weights of the first convolutional layers of a ChromWaveNet model from file.
    :param directory: directory that stored the serialized ChromWaveNet object
    :return: a list of weights of the convolutional filters (and biases) of the first convolutional layers, eg. those
    whose name contains the string 'conv_0' - by construction these are the convolutional layers that take the
    Input layer as input.
    """
    if os.path.exists(os.path.join(directory)):
        m = ChromWaveNet()
        m.deserialize(os.path.dirname(directory),os.path.basename(directory))
        base_model = m.get_underlying_neural_network()
        l_names = m.layer_names()
        conv_weights_pretrained = [base_model.get_layer(s).get_weights() for s in l_names if 'conv_0' in s]
    else:
        print('Warning: model does not exists  ')
        conv_weights_pretrained = []

    return conv_weights_pretrained

def make_optimizer(optimizer, lr, momentum=0.0, decay=0.0, nesterov=False, epsilon=None,amsgrad=False):
    """
    Function to create a keras optimiser with given parameters.
    :param optimizer: Choose from 'sgd', 'adam', 'nadam' and 'rmsprop'
    :param lr: learning rate for the optimiser
    :param momentum: momentum for the optimiser, defaults to 0.0 (not applicable for 'rmsprop')
    :param decay: decay for the optimisers, defaults to 0.0
    :param nesterov: flat wehter nesterov momentum shoould be used in 'sgd', defaults to None
    :param epsilon: epsilon used for the optimisers, defaults to None. Not applicable to 'sgd'
    :param amsgrad: flag whether 'amsgrad' should be used for 'adam' optimiser
    :return: build keras optimiser
    """
    if optimizer == 'sgd':
        optim = sgd(lr, momentum, decay, nesterov)
    elif optimizer == 'adam':
        optim = adam(lr=lr, beta_1= 0.9, beta_2=0.999,epsilon=epsilon,decay=decay, amsgrad=amsgrad)
    elif optimizer == 'nadam':
        optim = nadam(lr=lr, beta_1= 0.9, beta_2=0.999,epsilon=epsilon, schedule_decay = decay )
    elif optimizer == 'rmsprop':
        optim = RMSprop(lr=lr, epsilon=epsilon, decay=decay)
    else:
        raise ValueError('Invalid config for optimizer.optimizer: ' + optimizer)
    return optim




class ChromWaveNet:
    """
    Object that wraps Keras Wavenet style neural network objects, allowing the tuning of parameters without a lot of
    packing / unpacking.

    Additionally it provides functions for serialization, training, predicting, in silico mutagenesis, and saliency maps

    # Properties

        n_channels
        n_output_features
        n_output_bins
        kernel_lengths
        pool_sizes
        conv_n_filters
        _pretrained_models
        _pretrained_kernel_lengths
        _pretrained_conv_n_filters
        _pool_pretrained_conv
        _train_pretrained_conv
        inject_pwm_dir
        inject_pwm_selection
        inject_pwm_include_rc
        n_stacks
        dilation_depth
        dilation_kernel_size
        use_skip_connections
        learn_all_outputs
        train_with_soft_target_stdev
        train_only_in_receptive_field
        use_bias
        momentum
        learning_rate
        weight_decay
        nesterov
        optimizer
        epsilon
        amsgrad
        dropout
        res_dropout
        res_l2
        final_l2
        minimum_batch_size
        max_epochs
        early_stopping_patience_fraction
        early_stopping
        reduceLR_rate
        batch_normalization
        keras_verbose
        class_weights
        penalise_misclassification
        penalties

    # Methods

        _compute_PFM_motif_detectors
        _load_conv_weights
        _plot_confusion_matrices
        _plot_training_history
        _run_training
        _save_chr_prediction
        build
        build_model
        change_input_length
        compute_confusion_matrices
        compute_saliency
        deserialize
        deserialize_json
        get_methods
        get_underlying_neural_network
        in_silico_mutagenesis
        inject_pwm
        layer_names
        load_keras_config
        load_model
        load_weights_from
        make_loss_function
        plot_all_training_history
        plot_confusion_matrices
        plot_model
        plot_motif_detectors
        plot_predicted_class_distributions
        plot_profile_predictions
        plot_training_history
        predict
        predict_smooth
        save_chr_predictions
        save_keras_config
        save_model
        save_weights_to
        serialize_json
        summary
        train
        validate


    """

    def __init__(self):
        # Apply what defaults we can here

        # Global network parameters

        # number of bases
        self.n_channels = 4
        # number of output profiles
        self.n_output_features = 1
        self.n_output_bins=[1]

        # Convolution layers
        self.kernel_lengths = [24]
        self.pool_sizes = []
        self.conv_n_filters = [60]

        # Pretrained convolutional layers
        self._pretrained_models = []
        self._pretrained_kernel_lengths =[]
        self._pretrained_conv_n_filters=[]
        self._pool_pretrained_conv=None
        self._train_pretrained_conv = None # Defaults to true for all pretrained layers
        #
        # Use of PWMs to initialise convolutional layers
        self.inject_pwm_dir = None # if not none first 2N filters of the first trainable conv layers ('conv_0') will be injected with pwms and their reverse complement from directory. these will not be automatically pooled!
        self.inject_pwm_selection = None # if none first conv layer will be injected with  ALL pwms from directory
        self.inject_pwm_include_rc = False

        # Dilations and residual blocks
        self.n_stacks = 2
        self.dilation_depth =9
        self.dilation_kernel_size = 2
        self.use_skip_connections = False
        self.learn_all_outputs = True
        self.train_with_soft_target_stdev = False
        self.train_only_in_receptive_field = False
        self.use_bias = False

        # some optimiser hyper-parameters
        self.momentum = 0.9
        self.learning_rate = 0.001
        self.weight_decay = 0.0
        self.nesterov = True
        self.optimizer = 'adam'
        self.epsilon=0.001
        self.amsgrad = False

        # some training-related architecture hyper-parameters
        self.dropout = 0.3
        self.res_dropout = 0.1
        self.res_l2 = 0.0
        self.final_l2 = 0.0

        # some training hyper-parameters
        self.minimum_batch_size = 32
        self.max_epochs = 100
        self.early_stopping_patience_fraction = 3
        self.early_stopping = True
        self.reduceLR_rate = 0.1

        # whether to apply batch normaliseation after each convolutional layer and residual block
        self.batch_normalization = False
        self.keras_verbose = 1  # how much verbose to print during training - see keras doc

        self.class_weights = None  # if class weights should be used (passed from runtimedataset object)

        # is misclassification to be penalised
        self.penalise_misclassification = False
        self.penalties = [None]

        self._neural_network = None


    def build_model(self):
        """
        This builds the keras ChromWaveNet model,  the first convolutional layers taking the input layer as input are named
        as follows
            1. 'conv_0_pretrained': If pre-trained convolutional layers are to be used a convolutional layer with the
            specified number of filters and kernel size is added - this will be populated with the pre-trained weights
            later. These layers can be frozen so that they are not further trained, in this case a '_frozen' is added
            to the layer name
            2. 'conv_0': Another trainable convolutional layer with given number of filters and kernel lengths as
            spedified by self.conv_n_filters and self.kernel_lengths.
         The model is stored in self._neural_network with the specified parameters

        Parameters used
        ----------------
        Residual Block:
        self.n_stacks: number of stacks of dilated convolutions (made up of self.dilation_depth number of resdiual blocks)
        self.dilation_depth: number of residual blocks to be stacked
        self.res_l2: l2 regularisation parameter of the convolutional layers in the residual block
        self.dilation_kernel_size: kernel size of the dilated convolusions in the residual block
        self.use_bias: binary flag whether to use a bias term in the convolutional layers in the residual block
        self.res_dropout: dropout applied after each residual block

        Pretrained models:
        self._train_pretrained_conv: should pretrained conv layers be trained or frozen? Defaults to training all layers.
        self._pretrained_conv_n_filters: number of pretrained convolutional filter
        self._pretrained_kernel_lengths: kernel size of pre-trained convolutional filter
        self._pool_pretrained_conv: pool size if pre-trained layer should be pooled (or None)

        Other:
        self.batch_normalization: should batch normalisation be performed?
        self.conv_n_filters: number of convolutional filters
        self.kernel_lengths: kernel lengths of convolutional filters
        self.dropout: dropout rate outside of residual block

        :param self

        :return: keras model

        """
        def residual_block(x,i,s):
            original_x = x
            # total_nb_filters=nb_filters+nb_pretrained_filters+nuc_nb_filters
            total_nb_filters = x.shape[-1]

            tanh_out = layers.convolutional.Conv1D(total_nb_filters, kernel_size=self.dilation_kernel_size, dilation_rate=2 ** i,
                                                   padding='same',
                                                   use_bias=self.use_bias,
                                                   name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh',
                                                   kernel_regularizer=regularizers.l2(self.res_l2))(x)
            sigm_out = layers.convolutional.Conv1D(total_nb_filters, kernel_size=self.dilation_kernel_size, dilation_rate=2 ** i,
                                                   padding='same',
                                                   use_bias=self.use_bias,
                                                   name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid',
                                                   kernel_regularizer=regularizers.l2(self.res_l2))(x)

            x = Multiply(name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])
            res_x = layers.convolutional.Conv1D(total_nb_filters, kernel_size=1, padding='same', use_bias=self.use_bias,
                                                kernel_regularizer=regularizers.l2(self.res_l2))(x)
            skip_x = layers.convolutional.Conv1D(total_nb_filters, kernel_size=1, padding='same', use_bias=self.use_bias,
                                                 kernel_regularizer=regularizers.l2(self.res_l2))(x)

            res_x = Add()([original_x, res_x])

            return res_x, skip_x

        # in keras conv1d the channels need to go last.
        input = layers.Input(shape=(self.input_nodes_number, self.n_channels), name='input_part')
        out = input
        initial_conv_layers = []

        # ADDING the pretrained filters

        for i in range(len(self._pretrained_conv_n_filters)):
            nb_filters = self._pretrained_conv_n_filters[i]
            kernel_length = self._pretrained_kernel_lengths[i]
            if nb_filters > 0 and kernel_length > 0:
                layer_name = 'conv_'+str(i)+'_pretrained'
                if not self._train_pretrained_conv[i]:
                    layer_name = layer_name + '_frozen'
                conv_out= layers.convolutional.Conv1D(nb_filters, kernel_length, dilation_rate=1, padding='same',
                                                        name=layer_name, trainable=self._train_pretrained_conv[i], activation=None)(out)

                if self.batch_normalization:
                    conv_out=layers.BatchNormalization()(conv_out)
                conv_out = layers.Activation('relu')(conv_out)
                if self._pool_pretrained_conv[i] is not None:
                    lR = layers.Permute((2, 1))(conv_out)
                    if self._pool_pretrained_conv[i] in {'global', 'global_max'}:
                        pool =  keras.layers.pooling.MaxPooling1D(padding='same', pool_size=nb_filters)(lR) # (self._layers[-1][1])))
                    elif self._pool_pretrained_conv[i] == 'global_avg':
                        pool = keras.layers.pooling.AveragePooling1D(padding='same', pool_size=nb_filters)(lR)
                    elif self._pool_pretrained_conv[i] not in {'global', 'global_max', 'global_avg'}:
                        pool = layers.MaxPooling1D(padding='same', pool_size=self._pool_pretrained_conv[i])(lR)
                    conv_out = layers.Permute((2, 1))(pool)

                initial_conv_layers.append(conv_out)


        # adding other filters
        for i in range(len(self.conv_n_filters)):
            nb_filters=self.conv_n_filters[i]
            kernel_length=self.kernel_lengths[i]
            if nb_filters > 0 and kernel_length>0:

                conv_out= layers.convolutional.Conv1D(nb_filters, kernel_length, dilation_rate=1, padding='same',
                                                    name='conv_'+str(i), activation=None)(out)

                if self.batch_normalization:
                    conv_out=layers.BatchNormalization()(conv_out)

                conv_out = layers.Activation('relu')(conv_out)

                if len(self.pool_sizes) > i and self.pool_sizes[i] is not None:
                    lR = layers.Permute((2, 1))(conv_out)
                    if self.pool_sizes[i] in {'global', 'global_max'}:
                        pool =  keras.layers.pooling.MaxPooling1D(padding='same', pool_size=nb_filters)(lR) # (self._layers[-1][1])))
                    elif self.pool_sizes[i] == 'global_avg':
                        pool = keras.layers.pooling.AveragePooling1D(padding='same', pool_size=nb_filters)(lR)
                    elif self.pool_sizes[i] not in {'global', 'global_max', 'global_avg'}:
                        if isinstance(self.pool_sizes[i], int) and self.pool_sizes[i]>1:
                            pool = layers.MaxPooling1D(padding='same', pool_size=self.pool_sizes[i])(lR)
                    conv_out = layers.Permute((2, 1))(pool)
                initial_conv_layers.append(conv_out)

        if len(initial_conv_layers)>1:
            out = layers.concatenate(initial_conv_layers)
        else:
            out = conv_out

        if self.n_stacks > 0:
            skip_connections = []
            for s in range(self.n_stacks):
                for i in range(0, self.dilation_depth + 1):
                    out, skip_out = residual_block(out,i,s)
                    out = layers.Dropout(self.res_dropout)(out)
                    skip_connections.append(skip_out)

            if self.use_skip_connections:
                out = Add()(skip_connections)

        if self.batch_normalization:
            out = layers.BatchNormalization()(out)

        out = layers.Activation('relu')(out)
        shared_out = layers.Dropout(self.dropout, name ='last_shared_layer')(out)
        out_list = []

        for i in range(self.n_output_features):

            out = layers.Convolution1D(self.n_output_bins[i], kernel_size=1, padding='same',activation=None,
                                       kernel_regularizer=regularizers.l2(self.final_l2))(shared_out)
            if self.batch_normalization:
                out = layers.BatchNormalization()(out)

            out = layers.Activation('relu')(out)
            out = layers.Dropout(self.dropout)(out)

            out = layers.Convolution1D(self.n_output_bins[i], kernel_size=1, padding='same', activation=None)(out)
            if not self.learn_all_outputs:
                raise DeprecationWarning(
                    'Learning on just all outputs is wasteful, now learning only inside receptive field.')
                out = layers.Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(
                    out)  # Based on gif in deepmind blog: take last output?

            out = layers.Activation('softmax', name="output_softmax_" + str(i))(out)
            out_list.append(out)

        model = Model(input, out_list)

        receptive_field = utils.compute_receptive_field(self.dilation_depth, self.dilation_kernel_size,self.n_stacks)
        #
        print('Receptive Field: %d ' % receptive_field)
        return model



    def validate(self):
        """
        Validates correctness of the passed parameters

        Parameters used
        ----------------
        self.n_channels
        self.input_nodes_number

        self.conv_n_filters
        self.kernel_lengths

        :return: boolean flag whether parameters are valid
        """
        if not self.n_channels or not self.input_nodes_number:
            return False

        if len(self.conv_n_filters) != len(self.kernel_lengths):
            return False


        if not self.n_output_features == len(self.n_output_bins):
            return False

        if not len(self.penalties) == self.n_output_features:
            return False




        return True



    def make_loss_function(self, _output_bins,current_penalties,_class_weights):

        """
        Makes a bespoke loss function that can use both the penalties and class weights to adjust the
        usual categorical crossentropy along the sequence to assign a higher cost if a class is misclassified
        over other classes - if these options are chosen. Class weights are usually passed from the RuntimeDataset
        object; these adjust the cross-entropy loss function so that classification of rarer classes has as much
        influence on the loss as the frequently occurring classes. This is particularly important in the case of highly
        imbalanced classes. If self.penalise_misclassification is True, the weight in the array current_penalties will
        in addition be applied to the loss function. The higher the weight for a class, the more weight is assigned to
        the prediction of that class in the loss function.

        Parameters used
        ----------------
        self.penalise_misclassification: whether misclassifcation should be further penalised

        :param _output_bins: number of output bins of the profile to be predicted
        :param current_penalties: array of current penalties to be applied if self.penalise_misclassification is True -
        if None, these will be replaced by a linearly increasing weights for high and low classes.
        :param _class_weights: class weights to adjust the loss function
        :return: w_categorical_crossentropy: The weighted categorical crossentropy loss function
        :return: w_array: The weight array applied. If self.penalise_misclassification is False this is an all-1-array.
        """

        if self.penalise_misclassification:
            if current_penalties is not None:
                w_array = current_penalties
            else:
                w_array = numpy.ones((_output_bins, _output_bins))
                z = int(numpy.ceil(_output_bins / 2.))
                w_array[:z, -z:] = numpy.logspace(1, 1.2, num=z)
                w_array[-z:, :z] = numpy.logspace(1.2, 1, num=z)

        else:
            w_array = numpy.ones((_output_bins, _output_bins))


        def w_categorical_crossentropy(y_true, y_pred):
            nb_cl = len(w_array)
            final_mask = K.zeros_like(y_pred[:, :, 0])
            y_pred_max = K.max(y_pred, axis=-1)
            y_pred_max = K.expand_dims(y_pred_max, 2)
            # returns boolean array of shape y_pred that has true for the predicted class
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                # whenever c_p eaquals c_t then multiply by weight
                final_mask += (
                    K.cast(w_array[c_t, c_p], K.floatx()) * K.cast(y_pred_max_mat[:, :, c_p], K.floatx()) * K.cast(
                        y_true[:, :, c_t], K.floatx()))

            def partial_loss(y_true, y_pred):
                epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
                return - tf.reduce_sum(tf.multiply(y_true * tf.math.log(y_pred), _class_weights))

            return partial_loss(y_true, y_pred) * final_mask

        return w_categorical_crossentropy, w_array

    def inject_pwm(self, pwm_list):
        """
        Injects position weight matrices into the convolutional layer 'conv_0' centered at midpoint of the kernel.

        Parameters used
        ----------------
        self.inject_pwm_include_rc: whether reverse complement should be injected as well. Note that adding the reverse
        complement may cause OOM errors
        :param pwm_list: list of PWMs as numpy arrays
        :return:
        """
        layer_names = [l.name for l in self._neural_network.layers]
        conv_layer_index = layer_names.index('conv_0')
        conv_layer = self._neural_network.layers[conv_layer_index]
        conv_weights = conv_layer.get_weights()
        w=conv_weights[0].shape[0]

        if self.inject_pwm_include_rc:
            assert conv_layer.filters >= 2 * len( pwm_list), 'Error: too many PWMs and not enough trainable filters provided!'
        else:
            assert conv_layer.filters >= len( pwm_list), 'Error: too many PWMs and not enough trainable filters provided!'

        for i in range(len(pwm_list)):
            pwm = pwm_list[i]
            pwm = numpy.log2(pwm / 0.25)
            pwm_length = len(pwm)
            pwm_start = w / 2 - pwm_length / 2
            pwm_stop = pwm_start + pwm_length
            pwm_r = pwm[::-1, ::-1]
            if pwm_start <0:
                pwm = pwm[-pwm_start:]
                pwm_r = pwm_r[-pwm_start:]
                pwm_start = 0
            if pwm_stop >w:
                pwm = pwm[0:(w)]
                pwm_r = pwm_r[0:(w)]
                pwm_stop = w
            # set the convolutional weights
            conv_weights[0][:, :, i] = 0
            conv_weights[0][pwm_start:pwm_stop, 0:4, i] = pwm[::-1]
            # set bias to 0
            conv_weights[1][i] = 0
            if self.inject_pwm_include_rc:
                conv_weights[0][:, :, -i] = 0
                conv_weights[0][pwm_start:pwm_stop, 0:4, -i] = pwm_r[::-1]
                conv_weights[1][-i] = 0

        conv_layer.set_weights(conv_weights)


    def _build(self):
        """
        This builds the keras ChromWaveNet model, and subsequently
            1. loads and sets the weights for the pretrained convolutional layers from pretrained models if
            len(self._pretrained_models)>0
            2. injects PWMs into the traineable convolutional layer if self.inject_pwm_dir is not None
            3. computes the cross-entropy loss function for each output profile (stored in self.loss) using class
            weights and penalties stored in self.class_weights and self.penalties. Misclassifciation penalties will be
            applied if self.penalise_misclassification is True and self.penalties are updated.
            4. computes training metrics (stored in self.all_metrics) and generates the optimiser for training (stored
            in self.optim)
            5. updates callbacks with early stopping and reduce learning rate stored in self.callbacks

        Parameters
        ----------
        self._pretrained_models
        self._train_pretrained_conv
        self.inject_pwm_dir
        self.inject_pwm_selection
        self.pool_sizes
        self.penalise_misclassification:
        self.max_epochs
        self.early_stopping_patience_fraction
        self.optimizer
        self.penalties
        self.n_output_bins
        self.class_weights
        self.learning_rate
        self.momentum
        self.weight_decay
        self.nesterov
        self.epsilon
        self.amsgrad
        self.train_with_soft_target_stdev
        self.train_only_in_receptive_field
        self.early_stopping
        self.reduceLR_rate


        Parameters returned:
        --------------------
        self._neural_network
        self.penalties
        self.loss
        self.early_stopping_patience
        self.optim
        self.all_metrics
        self.callbacks

        # Calls Methods
        ----------
        self.build_model()
        self.inject_pwm() if self.inject_pwm is not None
        self.make_loss_function

        :param
        :return:
        """

        # build keras model
        self.validate()

        self._neural_network=self.build_model()
        layer_dict = dict([(layer.name, layer) for layer in self._neural_network.layers])

        # for pretrained layers set the weights and change the layer names
        for i in range(len(self._pretrained_models)):
            # layer 0 is input layer
            layer_name = 'conv_' +str(i) +'_pretrained'
            if not self._train_pretrained_conv[i]:
                layer_name = layer_name + '_frozen'

            layer_dict[layer_name].set_weights(self._pretrained_models[i])

        # if PWMs are to be injected in the convolutional layer do this here
        if self.inject_pwm_dir is not None:
            if len(self.pool_sizes) ==0:
                print('Injecting pwms without pooling the first layer. Consider to set model.pool_sizes = [2] to reduce paramters.')
            print('Loading pwm weights...')
            pwm_list=load_weights.load_scerTF_pwm_conv_weights(self.inject_pwm_dir, dbf_list=self.inject_pwm_selection)
            self.inject_pwm(pwm_list)

        # making the loss function by defining a loss for each output profile.
        self.loss=[]
        penalties=[]
        for (_current_penalties, _output_bins, _class_weights) in zip(self.penalties, self.n_output_bins,
                                                                      self.class_weights):
                w_categorical_crossentropy, w_array = self.make_loss_function(_output_bins, _current_penalties,
                                                                          _class_weights)
                w_categorical_crossentropy.__name__ = 'w_categorical_crossentropy' + str(len(self.loss))
                self.loss.append(w_categorical_crossentropy)
                penalties.append(w_array)

        if self.penalise_misclassification:
            self.penalties = penalties

        # setting up training-specific parameters
        self.early_stopping_patience = self.max_epochs / self.early_stopping_patience_fraction
        self.optim = make_optimizer(optimizer=self.optimizer, lr=self.learning_rate, momentum=self.momentum,decay= self.weight_decay,nesterov= self.nesterov, epsilon=self.epsilon,amsgrad=self.amsgrad)

        self.all_metrics = [kmetrics.categorical_accuracy, utils.categorical_mean_squared_error, utils.mean_classification_pearson_correlation]

        self.callbacks=[]
        if self.early_stopping:
            self.callbacks.extend(
                [EarlyStopping(patience=self.early_stopping_patience, verbose=1)])
        if self.reduceLR_rate is not None:
            if not self.reduceLR_rate ==0:
                self.callbacks.extend([ReduceLROnPlateau(patience=self.early_stopping_patience / 2, cooldown=self.early_stopping_patience / 4, verbose=1, factor=self.reduceLR_rate, min_lr=0.000001)])







    def build(self, runtime_dataset, pretrained_models=None, weight_classes=False):
        """
        Sets main parameters such as self.n_channels, self.n_output_features,.. using runtime_datset. Loads weights of
        pretrained models (if pretrained_models is not None) into self._pretrained_models and sets
        self._pretrained_kernel_lengths, self._pretrained_conv_n_filters accordingly. If self._pool_pretrained_conv is
        not defined for all pretrained layers, the default 'global' is applied.

        Parameters
        ----------
        self.optim
        self.loss
        self.all_metrics
        self.preprocessing_params

        Parameters returned:
        --------------------
        self.n_channels
        self.input_nodes_number
        self.n_output_bins
        self.n_output_features
        self.penalties (if None returns list of None)
        self.class_weights
        self._pretrained_models
        self._pretrained_kernel_lengths
        self._pretrained_conv_n_filters
        self._pool_pretrained_conv
        self._train_pretrained_conv
        self.preprocessing_params

        # Calls Methods
        ----------------
        self._load_conv_weights
        self._build
        self._neural_network.compile

        :param runtime_dataset: RuntimeDataset object
        :param pretrained_models: Dictionary of directories to the pretrained models as strings, available keys:
        'keras', 'numpy', 'chromwave'. Defaults to None.
        :param weight_classes: boolean flag. Should predictions of classes be differently weighted depending on their frequency?
        Recomended for highly imbalanced lcasses. Defaults to False.
        :return:
        """
        # needing some information from the data before we can build the model...
        self.n_channels = runtime_dataset.train_x_data.shape[2]  # 4 - BASES
        self.input_nodes_number = runtime_dataset.train_x_data.shape[1]
        self.n_output_bins = runtime_dataset._output_bins
        self.n_output_features = len(self.n_output_bins)
        self.preprocessing_params = runtime_dataset.preprocessing_params

        #Taking care of penalties on class weights:
        if self.penalties is None:
            self.penalties = [None] * self.n_output_features
        else:
            assert len(self.penalties) == self.n_output_features, "Error: wrong number of penalty weights supplied."

        if weight_classes:
            self.class_weights = runtime_dataset.class_weight
        else:
            self.class_weights = [numpy.ones_like(c) for c in runtime_dataset.class_weight]

        # Taking care of pretrained layers, loading weights, setting trainable flags
        # if pretrained models were defined, load the weights and make according layers in the model
        weights = []
        n_conv_pretrained = []
        kernel_l_pretrained = []

        if pretrained_models is not None:
            n_models = sum([len(m) if isinstance(m, list) else 1 for m in pretrained_models.values()])
            i=0
            for key, item in pretrained_models.items():
                if not isinstance(item, list):
                    item=[item]
                if len(item)>0:
                    model_type = key
                    for model_path in item:
                        if numpy.mod(i,10)==0:
                            print('Loading pre-trained model weights ' +str(i) + '/' + str(n_models))
                        w = self._load_conv_weights(model_path, model_type=model_type)
                        # if w[0] is not an array, it's a list or tuple of arrays:
                        if not hasattr(w[0], 'shape'):
                            weights.extend(w)
                            for x in w:
                                kernel_l_pretrained.append(x[0].shape[0])
                                n_conv_pretrained.append(x[0].shape[-1])
                        else:
                            weights.append(w)
                            kernel_l_pretrained.append(w[0].shape[0])
                            n_conv_pretrained.append(w[0].shape[-1])
                        i=i+1

        self._pretrained_models = weights
        self._pretrained_kernel_lengths = kernel_l_pretrained
        self._pretrained_conv_n_filters = n_conv_pretrained

        # Should pretrained convolutional layers be pooled? defaulting to pooling pretrained models
        if len(n_conv_pretrained)>0 and self._pool_pretrained_conv is None:
            print('Pooling not specified for pretrained models. Defaulting to global max pooling pretrained models.')
            self._pool_pretrained_conv = ['global'] * len(self._pretrained_models)

        elif self._pool_pretrained_conv is not None and len(self._pool_pretrained_conv)<len(self._pretrained_models):
                print('Defaulting to global max pooling for remaining non-specified pretrained models')
                self._pool_pretrained_conv.extend(['global'] * (len(self._pretrained_models)-len(self._pool_pretrained_conv)))

        # should pretrained layers be frozen or ruther trained? Defaults to training all pretrained layers
        if not self._train_pretrained_conv:
            if len(self._pretrained_conv_n_filters)>0:
                print('Defaulting to training all pretrained layers. These layers will be called conv_i_pretrained.')
            self._train_pretrained_conv=[True]*len(self._pretrained_conv_n_filters)
        else:
            if(isinstance(self._train_pretrained_conv,list)):
                if len(self._train_pretrained_conv)<len(self._pretrained_conv_n_filters):
                    if self._train_pretrained_conv[0]:
                        print('Defaulting to training all pretrained layers.These layers will be called conv_i_pretrained.')
                        self._train_pretrained_conv = [True] * len(self._pretrained_conv_n_filters)
                    else:
                        print('Defaulting to freezing and not training any pretrained layers. These layers will be called conv_i_pretrained_frozen.')
                        self._train_pretrained_conv = [False] * len(self._pretrained_conv_n_filters)
            else:
                if self._train_pretrained_conv:
                    print('Defaulting to training all pretrained layers. These layers will be called conv_i_pretrained.')
                    self._train_pretrained_conv = [True] * len(self._pretrained_conv_n_filters)
                else:
                    print('Defaulting to freezing and not training any pretrained layers. These layers will be called conv_i_pretrained_frozen.')
                    self._train_pretrained_conv = [False] * len(self._pretrained_conv_n_filters)



        self._build()

        self._neural_network.compile(optimizer=self.optim, loss=self.loss, metrics=self.all_metrics)

    def _run_training(self, runtime_dataset,run_dir,checkpoint_dir , pretrained_models=None, weight_classes=False,
                      rebuild=True , save=True):

        """
        Builds keras model if rebuild is True or if model does not exist yet. Extends callback with checkpoints.
        Saves the initial net and the setup json as well as a model summary if save is True. Runs training with specified
        number of max epochs. After training saves best model as keras model to file (h5 format) and saves its weights
        (h5 format) and saves the keras model as json. Saves the training history object as pickle file.

        Parameters
        ----------
        runtime_dataset.train_x_data
        runtime_dataset.train_y_data
        runtime_dataset.val_x_data
        runtime_dataset.val_y_data
        self.callbacks
        self.keras_verbose
        self.max_epochs


        Parameters returned:
        --------------------
        self.training_history
        self._neural_network

        # Calls Methods
        ----------------
        self._neural_network.save
        self.serialize_json
        self._neural_network.summary
        self._neural_network.fit
        self.save_model
        self.save_weights_to
        self._neural_network.to_json()

        :param runtime_dataset: RuntimeDataset object
        :param run_dir: string - path to run directory
        :param checkpoint_dir: string - path to checkpoint directory.
        :param pretrained_models: Dictionary of directories to the pretrained models asstrings, available keys:
        'keras', 'numpy', 'chromwave'. Defaults to None.
        :param weight_classes: boolean flag. Should predictions of classes be differently weighted depending on their frequency?
        :param rebuild: boolean flag if model should be rebuild if self._neural_network is not None, or if exisiting
        network should be continued to be trained.
        :param save: boolean flag if True saves the initial net as h5 file and the setup json as well as a model summary in run_dir
        :return:
        """

        if self.get_underlying_neural_network() is None or rebuild:
            self.build(runtime_dataset,  pretrained_models, weight_classes)
        else:
            self._neural_network.compile(optimizer=self.optim, loss=self.loss, metrics=self.all_metrics)


        self.callbacks.extend([
            ModelCheckpoint(os.path.join(checkpoint_dir, 'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5'),
                                save_best_only=True),
                CSVLogger(os.path.join(checkpoint_dir, 'history.csv')),
        ])

        if save:
            self._neural_network.save(os.path.join(run_dir, 'InitializedWaveNet.h5'))
            self.serialize_json(output_filepath=os.path.join(run_dir, 'WaveNet_Setup.json'))
            # save the model summary to file
            filename = os.path.join(run_dir, 'Modelsummary.txt')
            with open(filename, 'w') as fh:
                # Pass the file handle in as a lambda function to make it callable
                self._neural_network.summary(print_fn=lambda x: fh.write(x + '\n'))

        # keras training run
        history = self._neural_network.fit(x=runtime_dataset.train_x_data,
                            y=runtime_dataset.train_y_data,
                            epochs=self.max_epochs,
                            validation_data=(runtime_dataset.val_x_data, runtime_dataset.val_y_data),
                            callbacks=self.callbacks,
                            verbose=self.keras_verbose,
                            )
        self.training_history = history.history
        print("Saving model to file ... ")
        # save keras model as BestWaveNet.h5 in run)dir
        self.save_model(run_dir, 'BestWaveNet')

        # ave training history as pickle
        with open(os.path.join(run_dir, 'trainHistoryDict.pkl'), 'wb') as file_pi:
            pickle.dump(self.training_history, file_pi)

        # save keras model as json
        model_json = self._neural_network.to_json()
        with open(os.path.join(run_dir, "model.json"), "w") as json_file:
            json_file.write(model_json)

        # # serialize weights to HDF5 to BestWaveNet.weights
        self.save_weights_to(run_dir, "BestWaveNet")


    def train(self, runtime_dataset, filesystem, pretrained_models=None, weight_classes=False ,rebuild=True):

        """
        This trains the keras ChromWaveNet model and subsequently prints out the scores of the trained model evaluated on
        the training, test and validation data. Training history is plotted and motif filter computed and saved.
        Predictions fo all sequences stored in the genome_data in the RuntimeDataset object are saved.

        Calls Methods
        ----------------
        self._run_training
        self._neural_network.evaluate
        self.plot_all_training_history
        self._compute_PFM_motif_detectors
        self.save_chr_predictions

        :param runtime_dataset: RuntimeDataset object holding the data
        :param filesystem: FileSystem object
        :param pretrained_models: Dictionary of directories to the pretrained models asstrings, available keys:
        'keras', 'numpy', 'chromwave'. Defaults to None.
        :param weight_classes: boolean flag. Should predictions of classes be differently weighted depending on their frequency?
        :param rebuild: boolean flag if model should be rebuild if self._neural_network is not None, or if exisiting
        network should be continued to be trained.
        :return:
        """

        self._run_training(runtime_dataset, filesystem.get_output_directory(),filesystem.get_checkpoint_directory(), pretrained_models=pretrained_models, weight_classes=weight_classes ,rebuild=rebuild, save=True )


        print('Performance of model on trainings data: ')
        print(self._neural_network.evaluate(runtime_dataset.train_x_data,runtime_dataset.train_y_data))
        print('Performance of model on test data: ')
        print(self._neural_network.evaluate(runtime_dataset.test_x_data,runtime_dataset.test_y_data))
        print('Performance of model on validation data: ')
        print(self._neural_network.evaluate(runtime_dataset.val_x_data,runtime_dataset.val_y_data))

        print('Plotting training history...')
        self.plot_all_training_history(runtime_dataset, filesystem.get_output_directory())

        print('Computing motif detectors ...')
        self._compute_PFM_motif_detectors(runtime_dataset,filesystem.get_output_directory())

        print('Saving all chr predictions... ')
        self.save_chr_predictions(runtime_dataset,filesystem.get_output_directory())


    def plot_all_training_history(self,runtime_dataset, out_dir):
        """

        Plots all available diagnostic plots: loss and metrics vs training epochs, distributions of predicted and
        observed classes, confusion matrices and worst and best predictions measured by Pearson Correlation.

        :param runtime_dataset: RuntimeDataset holding the input data.
        :param outdir: output directory for the plots.
        :return:
        """
        if not os.path.exists(os.path.join(out_dir, 'plots')):
            os.makedirs(os.path.join(out_dir, 'plots'))
        if not os.path.exists(os.path.join(out_dir, 'plots','diagnostics')):
            os.makedirs(os.path.join(out_dir, 'plots','diagnostics'))

        plot_dir=os.path.join(out_dir, 'plots')
        diganostic_plot_dir=os.path.join(plot_dir,'diagnostics')

        if hasattr(self, 'training_history'):
            if self.training_history is not None:
                self.plot_training_history(diganostic_plot_dir)

        y_pred = self.predict(runtime_dataset.train_x_data)
        predictions = [y.argmax(axis=2) for y in y_pred]
        true = [y.argmax(axis=2) for y in runtime_dataset.train_y_data]

        print('Plotting distribution of class predictions, confusion matrices, and  best/worst predictions profiles...')
        self.plot_predicted_class_distributions(predictions, true,diganostic_plot_dir,what='training')
        self.plot_profile_predictions(predictions, runtime_dataset.train_y_data_smooth,plot_dir, what='training')

        cm=[confusion_matrix(numpy.hstack(x), numpy.hstack(y)) for x, y in zip(true, predictions)]
        self._plot_confusion_matrices(cm, out_dir = plot_dir, what='training', normalize = True)

        y_pred = self.predict(runtime_dataset.test_x_data)
        predictions = [y.argmax(axis=2) for y in y_pred]
        self.plot_profile_predictions(predictions, runtime_dataset.test_y_data_smooth,plot_dir, what='test')
        true = [y.argmax(axis=2) for y in runtime_dataset.test_y_data]
        self.plot_predicted_class_distributions(predictions, true,diganostic_plot_dir, what='test')
        cm=[confusion_matrix(numpy.hstack(x), numpy.hstack(y)) for x, y in zip(true, predictions)]
        self._plot_confusion_matrices(cm, out_dir = plot_dir, what='test', normalize = True)


        y_pred = self.predict(runtime_dataset.val_x_data)
        predictions = [y.argmax(axis=2) for y in y_pred]
        self.plot_profile_predictions(predictions, runtime_dataset.val_y_data_smooth, plot_dir, what='validation')

        true = [y.argmax(axis=2) for y in runtime_dataset.val_y_data]
        self.plot_predicted_class_distributions(predictions, true,diganostic_plot_dir, what='validation')

        cm=[confusion_matrix(numpy.hstack(x), numpy.hstack(y)) for x, y in zip(true, predictions)]
        self._plot_confusion_matrices(cm, out_dir = plot_dir, what='validation', normalize = True)


    def predict(self,x_data):
        """
        Makes raw predictions for the provided input data.

        :param x_data: one-hot-encoded sequence data. Must be of same lenght (shape[-1]) of self.input_node_num
        :return: list of predictions in one-hot encoded format for each output profile
        """
        y_pred= self._neural_network.predict(x_data)
        # if only one output feature, default is that the prediciton is not a list. make it list in this case
        if self.n_output_features == 1:
            y_pred=[y_pred]
        return(y_pred)

    def predict_smooth(self,x_data):
        """
        Returns smooth prediction profiles for each output profile for provided input data. The

        :param x_data: one-hot-encoded sequence data.
        :param runtime_dataset: RuntimeDataset object
        :return: list of smooth predicted profiles for each output profile
        """
        input_length = self.input_nodes_number
        changed_flag =False
        if not input_length== x_data.shape[1]:
            self.change_input_length()
            changed_flag=True
        y_pred = self.predict(x_data)
        predictions = [y.argmax(axis=2) for y in y_pred]

        smoothed_predictions = self.invert_discretizing(predictions)
        if changed_flag:
            self.change_input_length(input_length)
        return smoothed_predictions


    def invert_discretizing(self,y):
        """
        Inverts the discretization of a discretised array using smoothing.

        Parameters
        ----------
        self.preprocessing_params

        Calls
        -----
        self._smooth_discrete_data

        :param y: list of discretised numpy arrays
        :return: list of smooth numpy arrays.
        """
        y_smooth = [self._smooth_discrete_data(copy(seq), params) for (seq, params) in
         zip(y, self.preprocessing_params)]
        return y_smooth

    def _smooth_discrete_data(self,y,preprocessing_params):
        """
        smoothes a discretised numpy array

        :param y: discrete numpy array
        :param u: the parameter used to strecht the signal originally
        :return:
        """
        u = preprocessing_params['u']
        y = [numpy.array(int_to_float(seq, u)) for seq in y]
        y=numpy.vstack([numpy.array(gaussian_filter1d( seq, sigma=3,truncate=4.0)) for seq in y])
        return y

    def _plot_training_history(self, key, plot_title, plot_name, output_dir ):
        """
        Plots and item in self.training_history.

        :param key: string. What to plot, choose from self.training_history. keys()
        :param plot_title: string. title of plot.
        :param plot_name: string. name of plot
        :param output_dir: string. output directory
        :return:
        """
        plt.plot(self.training_history[key])
        if 'val_' + key in self.training_history:
            plt.plot(self.training_history['val_' + key])
        plt.title(plot_title)
        plt.ylabel(plot_title)
        plt.xlabel('epoch')
        if 'val_' + key in self.training_history:
            plt.legend(['train', 'test'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        # plt.show()
        plt.savefig(os.path.join(output_dir,
                                plot_name))
        plt.close()


    def plot_training_history(self, out_dir):
        """
        Plots the different metrics (accuracy, Pearson correlation) and loss vs training epochs.
        :param out_dir: string. output directory.
        :return:
        """
        if not hasattr(self,'training_history'):
            print('Nothing to plot')
            return
        else:
            # summarise history for Accuracy
            keys = [k  for k in self.training_history.keys() if 'categorical_accuracy' in k and not 'val' in k]
            for k in keys:
                self._plot_training_history(key=k, plot_title='categorical_accuracy', plot_name= "Training_History_Accuracy_" + str(keys.index(k)) + ".png", output_dir=os.path.join(out_dir))

            # summarise history for Pearson Correlation
            keys = [k  for k in self.training_history.keys() if 'mean_classification_pearson_correlation' in k and not 'val' in k]
            for k in keys:

                self._plot_training_history(key=k, plot_title='mean_classification_pearson_correlation',
                                                plot_name="Training_History_Pearson_Corr_" + str(keys.index(k)) + ".png",
                                                output_dir=os.path.join(out_dir))

            keys = [k  for k in self.training_history.keys() if 'loss' in k and not 'val' in k]
            for k in keys:

                self._plot_training_history(key=k, plot_title='loss',
                                                plot_name="Training_History_Loss_" + str(keys.index(k)) + ".png",
                                                output_dir=os.path.join(out_dir))




    def _save_chr_prediction(self, runtime_dataset, chr_name, directory):
        """
        Computes and saves prediction along a whole chromosome
        :param runtime_dataset: RuntimeDataset holding input data
        :param directory:string. output directory
        :param chr_name: string, chromosome name, must be in runtime_dataset.chr_info.
        :return:
        """
        changed_flag=False
        if chr_name=='chrM' and hasattr(runtime_dataset,'chrM_genome_data'):
            x = numpy.expand_dims(numpy.vstack(runtime_dataset.chrM_genome_data).transpose(), axis=0)
        else:
            i = runtime_dataset.chr_info.index(chr_name)
            assert i is not None, 'Error: chromosome  data not available in runtime_dataset'
            if not os.path.exists(directory):
                os.makedirs(directory)
            x = numpy.expand_dims(numpy.vstack(runtime_dataset.genome_data[i]).transpose(), axis=0)

        if self.get_underlying_neural_network().input_shape[1] is not None:
            input_length = self.input_nodes_number
            self.change_input_length()
            changed_flag=True

        y_p = self.predict(x)

        predictions = [y.argmax(axis=2) for y in y_p]

        smoothed_predictions =self.invert_discretizing(predictions)

        if hasattr(runtime_dataset, 'train_chr_info'):
            if chr_name in runtime_dataset.train_chr_info:
                numpy.save(os.path.join(directory, 'train_' + chr_name + '_predictions_profile.npy'),
                           numpy.vstack(smoothed_predictions))

            elif chr_name in runtime_dataset.test_chr_info:
                numpy.save(os.path.join(directory, 'test_' + chr_name + '_predictions_profile.npy'),
                           numpy.vstack(smoothed_predictions))
            elif chr_name in runtime_dataset.val_chr_info:
                numpy.save(os.path.join(directory, 'val_' + chr_name + '_predictions_profile.npy'),
                           numpy.vstack(smoothed_predictions))
            else:
                numpy.save(os.path.join(directory, chr_name + '_predictions_profile.npy'),
                           numpy.vstack(smoothed_predictions))
        else:
            numpy.save(os.path.join(directory, chr_name + '_predictions_profile.npy'),
                       numpy.vstack(smoothed_predictions))
        if changed_flag==True:
            self.change_input_length(input_length)



    def save_chr_predictions(self, runtime_dataset,out_dir,chr_name=None):
        """
        Saves prediction along a whole chromosome
        :param runtime_dataset: RuntimeDataset holding input data
        :param out_dir:string. output directory
        :param chr_name: string, chromosome name, must be in runtime_dataset.chr_info. Defaults to None in which case
        the predictions for all sequences in runtime_dataset.chr_info will be computed and saved.
        :return:
        """
        input_length = self.input_nodes_number
        self.change_input_length()

        directory = os.path.join(out_dir, 'data', 'predictions')
        if not os.path.exists(directory):
            os.makedirs(directory)
        if chr_name is None:
            print('plotting the predictions for all chromosomes')
            for chr_name in runtime_dataset.chr_info:
                self._save_chr_prediction(runtime_dataset, chr_name, directory)

        else:
            self._save_chr_prediction(runtime_dataset,chr_name,directory)


        self.change_input_length(input_nodes_number=input_length)




    def plot_predicted_class_distributions(self,predictions, true,output_dir, what='training'):
        """
        Plots distribution of predicted and observed classes
        :param predictions: list of arrays of profile predictions
        :param true: list of arrays of observed profiles
        :param out_dir: string. output directory
        :param what: choose from 'training', 'test', and 'validation
        :return:
        """
        for i in range(len(predictions)):
            plt.plot(numpy.unique(predictions[i].flatten(), return_counts=True)[0],
                     numpy.unique(predictions[i].flatten(), return_counts=True)[1])
            plt.plot(numpy.unique(true[i].flatten(), return_counts=True)[0],
                     numpy.unique(true[i].flatten(), return_counts=True)[1])
            plot_name='Class_counts_'+what+'_'+str(i)
            plt.title(plot_name)
            plt.ylabel('Class counts')
            plt.xlabel('Class')
            plt.legend(['predicted', 'true'], loc='upper left')
            # plt.show()
            plt.savefig(os.path.join(output_dir,
                                    plot_name))
            plt.close()

    def plot_profile_predictions(self,  predictions, true,  out_dir=None, what='training'):
        """
        Saves the prediction profiles of the best and worst sequences measured by Pearson correlation.
        :param predictions: list of arrays of profile predictions
        :param true: list of arrays of observed profiles
        :param out_dir: string. output directory
        :param what: choose from 'training', 'test', and 'validation
        :return:
        """
        idx = [utils.p_cor(t, p).argmax() for (t,p) in zip(true, predictions)]
        for i in idx:
            plot_name="WaveNet_prediction_"+what+"_maxCor_idx_" + str(idx.index(i))+ "_" +str(i)
            plot_title= 'Predicted vs observed Binding profiles - ' + what + ' set; min Correlation'
            plot.plot_profile_predictions(numpy.vstack([y[i] for y in true]), numpy.vstack([p[i] for p in predictions]), output_path= out_dir,
                                          plot_name=plot_name,
                                          plot_title=plot_title)


        idx = [utils.p_cor(t, p).argmin() for (t,p) in zip(true, predictions)]
        for i in idx:
            plot_name="WaveNet_prediction_"+what+"_minCor_idx_" + str(idx.index(i))+ "_" +str(i)
            plot_title= 'Predicted vs observed Binding profiles - ' + what + ' set; min Correlation'
            plot.plot_profile_predictions(numpy.vstack([y[i] for y in true]), numpy.vstack([p[i] for p in predictions]), output_path= out_dir,
                                          plot_name=plot_name,
                                          plot_title=plot_title)


    def _plot_confusion_matrices(self, cm, out_dir, what='training', normalize = False):
        """
        Saves plot of confusion matric to file
        :param cm: numpy array holding the confusion matrix
        :param out_dir: string. Output directory
        :param what: string. Choose from 'training','test' and 'validation'.
        :param normalize: boolean flag whether rows should be normalised to sum to 1.
        :return:
        """
        print('Plotting the confusion matrix for ' + what + ' set.....')
        for i in range(len(cm)):
            plot_name = 'Confusion matrix ' + what + ' ' +str(i)
            plot.plot_confusion_matrix(cm[i], normalize=normalize,
                                       title=plot_name, output_path=out_dir, plot_name=plot_name)


    def compute_confusion_matrices(self, runtime_dataset, what = 'training'):
        """
        Computes the confusion matrices between the true and observed output classes.
        :param runtime_dataset: RuntimeDataset object holding the input data.
        :param what: string. Choose from 'training', 'test' and 'validation'.
        :return:
        """
        print('Computing the confusion matrix for ' +what+' set.....')
        if what=='training':
            predictions= [numpy.vstack([x.argmax(axis=-1) for x in y]) for y in self.predict(runtime_dataset.train_x_data)]
            labels = [numpy.vstack([x.argmax(axis=-1) for x in y]) for y in runtime_dataset.train_y_data]

        elif what=='test':
            predictions = [numpy.vstack([x.argmax(axis=-1) for x in y]) for y in self.predict(runtime_dataset.test_x_data)]
            labels = [numpy.vstack([x.argmax(axis=-1) for x in y]) for y in runtime_dataset.test_y_data]

        else:
            predictions = [numpy.vstack([x.argmax(axis=-1) for x in y]) for y in self.predict(runtime_dataset.val_x_data)]
            labels = [numpy.vstack([x.argmax(axis=-1) for x in y]) for y in runtime_dataset.val_y_data]

        return [confusion_matrix(numpy.hstack(x), numpy.hstack(y)) for x, y in zip(labels, predictions)]



    def plot_confusion_matrices(self,runtime_dataset, out_dir, what='training', normalize=False):
        """
        Computes and plots confusion matrices for the selected dataset.
        :param runtime_dataset: RuntimeDataset holding the input data.
        :param out_dir: string. output directory
        :param what: choose from 'training', 'test' and 'validation'
        :param normalize: boolean flag if rows of confusion matrix should be normalised so they sum to 1.
        :return:
        """
        cm = self.compute_confusion_matrices(runtime_dataset, what = what)
        self._plot_confusion_matrices(cm,out_dir=out_dir, what=what, normalize = normalize)


    def summary(self):
        """

        :return:
        """
        return self._neural_network.summary()

    def plot_model(self, directory):
        """

        :param directory:
        :return:
        """
        kplot_model(self.get_underlying_neural_network(),to_file = directory)

    # Function to deserialize from best
    def deserialize(self, directory, file_name = 'BestWaveNet.h5'):
        """
        This function loads a genomic neural network from disk.
        Note the behaviour of this network is guaranteed to be identical to that of it's parent, except when mutated / rebuilt (with mutation) - as the state of the pseudo_rng used cannot be fully guaranteed.
        Parameters
        ----------

        Parameters returned:
        --------------------

        Calls Methods
        -------------
        :param directory: The source directory from which to deserialize the genomic neural network
        id: filename of the h5 file to load, choices
        :return:
        """

        if not os.path.isdir(directory):
            print('File path is not a directory.')
            return

        setup_file=os.path.join(directory,'WaveNet_Setup.json')
        if not os.path.isfile(setup_file):
            print('Setup file does not exist.')
            return

        self.deserialize_json(setup_file)


        model_file=os.path.join(directory,file_name)

        if not os.path.isfile(model_file):
            print('Model file does not exist.')
            return

        custom_objects = {'categorical_mean_squared_error': utils.categorical_mean_squared_error,
                          'mean_classification_pearson_correlation': utils.mean_classification_pearson_correlation}

        for i in range(len(self.loss)):
            custom_objects['w_categorical_crossentropy' + str(i)] = self.loss[i]

        id= str.split(file_name,'.')[0]
        self.load_model(directory, id=id, custom_object_dict=custom_objects)





    def change_input_length(self,input_nodes_number = None):
        """
        To be able to make predictions along sequences of different length than the input sequences used during training,
        the input length of the model has to be changed, this is achieved by temporarily storing the weights and then
        rebuilding the model with the chosen input length and finally loading the stored weights of the original model.

        Parameters:
        ----------
        self.optim
        self.loss
        self.all_metrics

        Parameters returned:
        --------------------
        self.input_nodes_number

        Calls Methods
        -------------
        self.get_underlying_neural_network().get_weights
        self._build
        self._neural_network.compile


        :param input_nodes_number:
        :return:
        """
        # returns the same network with the same weights but with chosen input nodes number. choose None for variable
        # input lenghts.

        # store weights
        weights = self.get_underlying_neural_network().get_weights()
        #
        self.input_nodes_number = input_nodes_number
        self._build()
        # set weights
        self.get_underlying_neural_network().set_weights(weights)
        self._neural_network.compile(optimizer=self.optim, loss=self.loss, metrics=self.all_metrics)

    def save_keras_config(self, output_filepath = None, id=""):
        """

        :param output_filepath:
        :param id:
        :return:
        """
        if self._neural_network:
            keras_params = kfb.to_dict_w_opt(self._neural_network)
            if output_filepath:
                utils.save_json(keras_params, os.path.join(output_filepath, id + '.keras_config' ))
        return keras_params


    def load_keras_config(self, source_path, id="", custom_object_dict=None):
        """
        :param source_path:
        :param id:
        :param custom_object_dict:
        :return:
        """
        keras_config=utils.load_json(os.path.join(source_path, id + '.keras_config' ))
        self._neural_network = kfb.model_from_dict_w_opt(keras_config, custom_objects=custom_object_dict)



    def save_model(self, output_path, id=""):
        """

        :param output_path:
        :param id:
        :return:
        """
        self._neural_network.save(os.path.join(output_path, id + ".h5"))

    def load_model(self,source_path, id="", custom_object_dict=None):
        """

        :param source_path:
        :param id:
        :param custom_object_dict:
        :return:
        """
        self._neural_network= keras.models.load_model(os.path.join(source_path, id + ".h5"),custom_objects=custom_object_dict)


    def load_weights_from(self, source_path, id=""):
        """

        :param source_path:
        :param id:
        :return:
        :return:
        """

        self._neural_network.load_weights(os.path.join(source_path, id + ".weights"))

    def save_weights_to(self, output_path, id=""):
        """

        :param output_path:
        :param id:
        :return:
        """

        self._neural_network.save_weights(os.path.join(output_path, id + ".weights"), overwrite=True)


    # Function used to build the current genomic neural net configuration from a json object
    # this is mainly only for information only. everything else is done with keras
    def deserialize_json(self, source):
        """
        This function loads the basic NN parameters from disk, but not fitted parameters.
        Parameters
        ----------

        Parameters returned:
        --------------------

        Calls Methods
        -------------
        :param source: the source json file
        :return:
        """

        if not os.path.isfile(source):
            print('Source file ' + source+ '  not found.')
            return
        json = utils.load_json(source)
        self.input_nodes_number = json["input_nodes_number"]
        self.n_channels = json["n_channels"]
        self.n_output_features =json["n_output_features"]
        self.n_output_bins=json["n_output_bins"]

        self.max_epochs = json["max_epochs"]
        self.minimum_batch_size = json["minibatch_size"]
        self.early_stopping_patience_fraction=json["early_stopping_patience_fraction"]

        # hyper parameters
        self.momentum = json["momentum"]
        self.learning_rate = json["learning_rate"]
        self.weight_decay = json["weight_decay"]
        self.nesterov = json["nesterov"]
        self.optimizer = json["optimizer"]
        self.epsilon=json["epsilon"]

        self.dropout = json["dropout"]
        self.res_dropout = json["res_dropout"]
        self.res_l2 = json["res_l2"]
        self.final_l2 = json["final_l2"]

        # Runtime Type
        self.penalise_misclassification = json["penalise_misclassification"]
        self.penalties=[numpy.array(p) for p in json["penalties"]]
        self.class_weights=[numpy.array(c) for c in json["class_weights"]]
        self.preprocessing_params = json["preprocessing_params"]
        self.batch_normalization=json['batch_normalization']

        # pretrained models
        self._pretrained_models = [list(map(numpy.array, m )) for m in json["_pretrained_models"]]
        self._pretrained_kernel_lengths =json["_pretrained_kernel_lengths"]
        self._pretrained_conv_n_filters=json["_pretrained_conv_n_filters"]
        self._pool_pretrained_conv = json['_pool_pretrained_conv']
        self._train_pretrained_conv = json['_train_pretrained_conv']
        # Convolution layers
        self.n_stacks = json["n_stacks"]
        self.dilation_depth =json["dilation_depth"]
        self.kernel_lengths = json["kernel_lengths"]
        self.conv_n_filters = json["conv_n_filters"]
        self.pool_sizes = json["pool_sizes"]
        self.use_skip_connections = json["use_skip_connections"]
        self.learn_all_outputs = json["learn_all_outputs"]
        self.use_bias = json["use_bias"]

        self.train_with_soft_target_stdev = json["train_with_soft_target_stdev"]
        self.train_only_in_receptive_field = json["train_only_in_receptive_field"]
        self.keras_verbose = json["keras_verbose"]

        if not isinstance(self.penalties, list):
            print('Penalties were not saved as list. Converting.. ')
            self.penalties=[self.penalties]

        # Trust the values we have been given - this overcomes reproducibility issues associated with pseudo-rng states
        self._build()



    # Function to convert a neural net to json for serialization
    def serialize_json(self, output_filepath=None):
        """
        Serialize the basic neural net parameters to file. It will not serialize fitted parameters
        Parameters
        ----------

        Parameters returned:
        --------------------

        Calls Methods
        -------------
        :param output_filepath: the file path to the output
        :return:
        """

        net_parameters = {
        "input_nodes_number": self.input_nodes_number,
        "n_channels":self.n_channels ,
        "n_output_features":self.n_output_features,
        "n_output_bins":self.n_output_bins ,

        "max_epochs":self.max_epochs ,
        "minibatch_size":self.minimum_batch_size ,
        "early_stopping_patience_fraction":self.early_stopping_patience_fraction,

        # hyper parameters
        "momentum":self.momentum ,
        "learning_rate":self.learning_rate,
        "weight_decay":self.weight_decay ,
        "nesterov":self.nesterov,
        "optimizer":self.optimizer,
        "epsilon":self.epsilon,

        "dropout":self.dropout ,
        "res_dropout": self.res_dropout,
        "res_l2":self.res_l2 ,
        "final_l2":self.final_l2 ,

        # # Runtime Type
        "penalise_misclassification":self.penalise_misclassification ,

        "penalties":[ utils.make_list(p) for p in self.penalties ],
        "class_weights":[utils.make_list(c) for c in self.class_weights],
         "preprocessing_params": self.preprocessing_params,
        "batch_normalization": self.batch_normalization,
        #
        #
        # # pretrained models
        "_pretrained_models":[ list(map(utils.make_list, m)) for m in self._pretrained_models],
        "_pretrained_kernel_lengths":self._pretrained_kernel_lengths,
        "_pretrained_conv_n_filters":self._pretrained_conv_n_filters,
        '_pool_pretrained_conv' : self._pool_pretrained_conv,
        '_train_pretrained_conv' :   self._train_pretrained_conv,
            #
        # Convolution layers
        "n_stacks":self.n_stacks ,
        "dilation_depth":self.dilation_depth ,
        "kernel_lengths":self.kernel_lengths ,
        "conv_n_filters":self.conv_n_filters ,
        "pool_sizes":self.pool_sizes ,
        "use_skip_connections":self.use_skip_connections ,
        "learn_all_outputs":self.learn_all_outputs ,
        "use_bias":self.use_bias ,

        "train_with_soft_target_stdev":self.train_with_soft_target_stdev,
        "train_only_in_receptive_field":self.train_only_in_receptive_field  ,
        "keras_verbose":self.keras_verbose

        }

        # Provide the ability to save this net
        if output_filepath:
            utils.save_json(net_parameters, output_filepath)

        return net_parameters

    def get_underlying_neural_network(self):
        """
        Returns the keras model.
        :return:
        """
        return self._neural_network

    def _compute_PFM_motif_detectors(self,  runtime_dataset,out_dir,percentile = 0):
        """
        Computes the motif detector position frequency matrices as in Alipanahi et al 2015 (DeepBind).
        If no _neural_network object attempts to desearialise from out_dir. Percentile can be chosen so that not all
        input sequences are used to compute the PFM, rather only those whose maximal activation is in the percentiles
        for the respective filter are used.

        :param runtime_dataset: RuntimeDataset object holding the input data.
        :param out_dir: string. Output directory
        :param percentile: float. Threshold for the maximal activation of a sequence to be included to compute PFM.
        Percentile which must be between 0 and 100 inclusive.
        :return:
        """
        print('Computing the PFMs of the motif detectors')

        if not os.path.exists(os.path.join(out_dir, 'motif-detectors')):
            os.makedirs(os.path.join(out_dir, 'motif-detectors'))

        if not self._neural_network:
            self.deserialize(out_dir)

        layer_dict = dict([(layer.name, layer) for layer in self._neural_network.layers])

        conv_layer_keys = [key for key in layer_dict.keys() if key.startswith('conv_')]

        for conv_layer_key in conv_layer_keys:
            conv_layer = self._neural_network.get_layer(conv_layer_key).output

            input_var=self._neural_network.input


            # getting positions of the max of the convolutional filters along each sequence: start_x is of shape (#training sequences, # filters of conv_layer)
            start = K.function([input_var], [K.argmax(conv_layer, axis=1)])
            start_x = numpy.asarray(start([runtime_dataset.train_x_data])[0])
            # getting the poition of the last sequence position that contributes to the argmax of the filter
            end_x = start_x + self._neural_network.get_layer(conv_layer_key).get_config()['kernel_size'][0]

            # we need compute the max and the percentile to exclude uninformative sequences from the motif plotting
            max = K.function([input_var], [K.max(conv_layer, axis=1)])
            max_list = numpy.asarray(max([runtime_dataset.train_x_data])[0])
            # computing threshold for the max to be above so that sequence is used for computation of motif detetorfiltres long
            t = numpy.percentile(max_list, percentile, axis=0)

            # if this was a three-channel attempt, then evaluate the missing row
            if (self.n_channels < 4):
                row = numpy.sum(runtime_dataset.train_x_data, axis=2)
                row = 1 - row.reshape(row.shape[0], row.shape[1], 1)
                x = numpy.transpose(numpy.column_stack((runtime_dataset.train_x_data, row)), (0,2,1))
            else:
                x = numpy.transpose(runtime_dataset.train_x_data, (0,2,1))
            current_motif_list = motif_viz.get_motif(x[0], start_x[0], end_x[0], max_list[0], self._neural_network.get_layer(conv_layer_key).get_config()['kernel_size'][0],
                                                     self._neural_network.get_layer(conv_layer_key).get_config()['filters'], thresh = t)
            for instance in range(1, x.shape[0]):
                list_of_motifs = motif_viz.get_motif(x[instance], start_x[instance], end_x[instance], max_list[instance],
                                                     self._neural_network.get_layer(conv_layer_key).get_config()[
                                                         'kernel_size'][0], self._neural_network.get_layer(conv_layer_key).get_config()['filters'], thresh = t)
                current_motif_list = list(map(add, current_motif_list, list_of_motifs))

            for i in range(len(current_motif_list)):
                if not all(current_motif_list[i].flatten() < 0.001):
                    # save the motif  unnormalised
                    motif_df = pandas.DataFrame(current_motif_list[i], index=['A', 'C', 'G', 'T'])
                    motif_df.to_csv(
                        os.path.join(out_dir, 'motif-detectors' ,conv_layer_key+'_motif_' + str(i) + '.csv'))

    def plot_model(self, directory=''):
        """
        Uses keras functionality to plot the model architecture to file 'model.png'
        :param directory: path to directory, defaults to working directory.
        :return:
        """
        print('Warning: layer names except first convolutional names will be changed if the model is plotted after calling self.change_input_length(). For correct layer names please plot directly after building or deserialising.')
        kplot_model(self._neural_network, to_file=os.path.join(directory, 'model.png'))

    def plot_motif_detectors(self,runtime_dataset,out_dir):
        """
        Calls an R script to visualise the computed motif detectors as seqLogo plots.
        :param runtime_dataset: RuntimeDataset object holding the input data.
        :param out_dir: string. Output directory.
        :return:
        """

        if not os.path.exists(os.path.join(out_dir, 'motif-detectors')):
            self._compute_PFM_motif_detectors(runtime_dataset,out_dir)
        # plot seq logo plots with R

        print('Plotting motif detectors with R')

        full_path_RScript = os.path.split(inspect.getfile(motif_viz))[0]

        command = 'Rscript'
        script = os.path.join(full_path_RScript,'motif_plot.R')
        args = [os.path.join(out_dir, 'motif-detectors'),
                os.path.join(out_dir, 'motif-detectors')
                ]
        cmd = [command, script] + args

        # plot PNGs of motifs (best done at the end to avoid re-loading R and libraries all the time)
        try:
            print(subprocess.check_output(cmd, universal_newlines=True))
        except subprocess.CalledProcessError as e:
            print('subprocess stdout output: \n', e.output)

        return

    def _load_conv_weights(self, model_path, model_type):
        """
        Loads weights of pretrained convolutional layers.

        :param model_path: string. directory to the weights file.
        :param model_type: string. choose from 'keras', 'chromwave', and 'numpy' to load weights from a keras model, a
        ChromWaveNet model or a numpy array. If numpy array is chosen the files 'conv_weights.npy' and 'conv_bias.npy'
        are expected in the directory.
        :return: a list of convolutional weights.
        """

        if model_type == 'keras':
            return load_weights.load_existing_keras_conv_weights(model_path)
        elif model_type == 'chromwave':
            return load_existing_wavenet_conv_weights(model_path)
        elif model_type == 'numpy':
            return numpy.load(os.path.join(model_path,'conv_weights.npy')),numpy.load(os.path.join(model_path,'conv_bias.npy'))
        else:
            print('Error, model type not implemented yet. Choose keras, chromwave or numpy only.')


    def in_silico_mutagenesis(self, sequence_data):
        """
        Computes the in silico mutagenesis scores for the sequence data and returns the results as a list of arrays
        ISM scores (the difference in predictions between base change and WT), the predictions for each base change, 
        the preactivations for each base change, and the difference of preactivations for each base change and WT. 
        For sample 0, first base, the predictions of all possible basechanges are in mutagenesis_scores[0][0,0,:,:]
        Parameters
        ----------

        Parameters returned:
        --------------------

        Calls Methods
        -------------
        :param sequence_data: numpy array of shape [1, self.input_nodes_number, num_bases]
        :return:  A list of arrays, each array is of shape [n_samples, seq_lengt, seq_length,num_classes, num_bases]

        """

        assert sequence_data.shape[0]==1, 'Function untested for more than one sequence'
        if not sequence_data.shape[1] == self.input_nodes_number:
            self.change_input_length(sequence_data.shape[1])
        # target layers are those layers that are input to the last softmax layers. retrieving them by first finding
        # the softmax layers for each output feature and then getting the input layers to those layers.
        target_layers = []
        mutagenesis_scores= []
        for i in range(self.n_output_features):
            target_layer=self.get_underlying_neural_network().get_layer('output_softmax_' + str(i))._inbound_nodes[0].inbound_layers
            x_mut = saliency.in_silico_mutagenesis(self.get_underlying_neural_network(), sequence_data,  target_layer,i)

            predictions_mut = x_mut[1]
            from copy import copy
            def smooth_predictions(arr):
                return (numpy.squeeze(self._smooth_discrete_data(copy(numpy.expand_dims(arr, 0)),
                                                              preprocessing_params=self.preprocessing_params[i]), 0))

            predictions_mut_smooth = numpy.apply_along_axis(smooth_predictions, 2, predictions_mut)
            pred_smooth = self.predict_smooth(sequence_data)[i]
            v = pred_smooth[numpy.newaxis, :, :, numpy.newaxis]
            smoothed_diff = predictions_mut_smooth - v

            mutagenesis_scores.append([smoothed_diff,x_mut[1],x_mut[2],x_mut[3]])
        return mutagenesis_scores


    def compute_saliency(self, sequence_data, min=None, max=None, directory=None, plot_name='saliency',input_layer_index=0):

        """
        Computes the saliency map for the sequence data and returns the results as a list of arrays (one for each of the output profiles).
        Parameters
        ----------

        Parameters returned:
        --------------------

        Calls Methods
        -------------
        :param sequence_data: numpy array of shape [1, self.input_nodes_number, num_bases]
        :param min: min of range of the sequence to compute the saliency scores for. Defaults to none in which case the
        saliency map for the whole sequence is computed
        :param max: max of range of the sequence to compute the saliency scores for. Defaults to none in which case the
        saliency map for the whole sequence is computed
        :param directory: string. if not None, a plot of the sequence scaled by saliency scores is saved as pdf into the directory
        :param plot_name: string. name of the plot
        :param input_layer_index:  integer. index of the input layer, will usually be 0

        :return: A list of arrays holding the saliency map scores for each output profile. Each array is of shape [1,self.input_nodes_number, num_bases]
        """

        if not sequence_data.shape[1] == self.input_nodes_number:
            print('Please pad the sequence(s) to length '+str(self.input_nodes_number)+'.  Do not change th input lenght of the model - it will not retain the correct gradients!')


        model = self.get_underlying_neural_network()

        scores = [saliency.compute_gradient(model,sequence_data,output_index, min=min, max=max,input_layer_index =input_layer_index) for output_index in range(self.n_output_features)]
        scores_data = [numpy.sum(s, axis=-1) for s in scores]
        scores_for_seq = [sequence_data * s[:, None] for s in scores_data]

        if directory is not None:
            from .vis import plot
            for i in range(len(scores_for_seq)):
                fig = plot.plot_weights(scores_for_seq[i], subticks_frequency=10, title='saliency for outindex ' + str(i))
                fig.savefig(os.path.join(directory, plot_name + '_out_index_' + str(i) + '_.pdf'), format='pdf')
                plt.close(fig)

        return scores_for_seq

    def layer_names(self):
        """
        Gets the layer names of the underlying Keras model.
        :return: A list of layer names of the underlying keras model
        """
        return([l.name for l in self.get_underlying_neural_network().layers])

    def get_methods(self, spacing=20):
        methodList = []
        for method_name in dir(self):
            try:
                if callable(getattr(self, method_name)):
                    methodList.append(str(method_name))
            except:
                methodList.append(str(method_name))
        processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
        for method in methodList:
            try:
                print(str(method.ljust(spacing)) + ' ' +
                      processFunc(str(getattr(self, method).__doc__)[0:90]))
            except:
                print(method.ljust(spacing) + ' ' + ' getattr() failed')