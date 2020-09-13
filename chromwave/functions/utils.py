import errno

import json
import os

import numpy as np
import pandas

import keras.backend as K
import tensorflow as tf

import pickle
import scipy

from collections import OrderedDict
import shutil

__all__ = [
    "float32",
    'load_csv',
    'save_json',
    'load_json',
    'reverse_complement',
    'bases_to_binary',
    'padding_dna',
    'log_uniform',
    'sqrt_uniform'
]


def mkdir_p(path):
    """
    Creates the directory path, recursively.
    http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def float32(k):
    return np.cast['float32'](k)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_json(data, filename):
    with open(filename, 'w') as data_file:
        json.dump(data, data_file, cls=NpEncoder)


def load_json(filename):
    try:
        with open(filename, 'r') as data_file:
            return json.load(data_file)
    except IOError as e:
        print('Unable to open file ' + filename)
        print(e)
    except ValueError as e:
        print('Unable to read JSON content in file ' + filename)
        print(e)


def load_csv(fname, cols=None):
    df = pandas.read_csv(os.path.expanduser(fname), sep='\t', header=None)  # load pandas dataframe
    if cols:  # get a subset of columns
        df = df[list(cols)]
    return df

def pickle_save(obj,  filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(filename):
    return pickle.load(open(filename,'rb'))

# softmax function for an np array
def softmax(w):
    w = np.array(w)

    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)

    return dist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))




def write_roman(num):

    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num > 0:
                roman_num(num)
            else:
                break

    return "".join([a for a in roman_num(num)])



def reverse_complement(seq, matrix_bases='ACGT', error_letter='X'):
    """
    Function to take DNA, represented as either a string of letters or a binary
    matrix, and return its reverse complement.

    Parameters
    ----------
    seq : str or array_like
        A string of DNA letters, mixed case permitted, or an n x 4 array of
        1s and 0s.
    matrix_bases : str, optional
        If seq is an array, this defines which column corresponds to which base.
    error_letter : str
        If seq is a string, this is the letter which will be returned in
        positions where the character does not correspond to a valid base.
    
    >>> reverse_complement('AACCTCGATGG')
    'CCATCGAGGTT'
    >>> reverse_complement(bases_to_binary('AACCTCGATGG'))
    'array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
       [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1]], dtype=int8)'
    >>> reverse_complement(np.array([[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]))
    array([[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 1]])
    """

    # Define complementary bases
    complement = {
        'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A',
        'a': 't', 'c': 'g', 'g': 'c', 't': 'a'
    }

    # if it's a string...
    if type(seq) is str:
        return ''.join(reversed([complement.get(base, error_letter) for base in seq]))

    # Otherwise, it's an np.ndarray...
    else:
        # ...which means we need to make a mini lookup table to work out how to
        # swap rows

        # First, reverse complement the matrix_bases string.
        matrix_bases_c = [
            complement.get(base, error_letter) for base in matrix_bases
            ]
        rowswap_to = []
        for i in range(len(matrix_bases)):
            # Add the position to send it to by looking up the position of the
            # ith base from matrix_bases in matrix_bases_rc
            rowswap_to.append(matrix_bases_c.index(matrix_bases[i]))
        # Finally, perform the swap, reverse and return the answer
        return seq[rowswap_to][:, ::-1]


def bases_to_binary(sequence, trim_start=0, trim_end=0, paddingL=0,paddingR=0 , matrix_bases='ACGT'):
    """
    Takes a string of DNA letters and turns them into an 4 x n array
    of 1s and 0s.

    Parameters
    ----------
    sequence : str
        A string of DNA letters, mixed case permitted.
    trim_start : int, optional
        Number of letters to remove from the start of the sequence, default 0.
    trim_end : int, optional
        Number of letters to remove from the end of the sequence, default 0.
    padding : int, optional
        Number of Ns (ie 4 0.25s) to add to the start and end of the sequence.
        should not be used for large datasets because it turns the array into a float array. should'nt be necessary
        with genomic net anyway because padding is done internally inside the net with a padding layer
    matrix_bases : str, optional
        If seq is an array, this defines which column corresponds to which base.
    
    >>> bases_to_binary('ACGTacgtX')
    array([[ 1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.25],
           [ 0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.25],
           [ 0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.25],
           [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.25]], dtype=float32)

    

    >>> bases_to_binary('ACGTacgtXx', trim_start = 2, trim_end = 3,
    ... padding = 1, matrix_bases = 'cgAT')
    array([[ 0.25,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.25],
           [ 0.25,  1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.25],
           [ 0.25,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.25],
           [ 0.25,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.25]])
    """
    binarised_bases =     np.array(
        [
            [    # trim start and end characters if necessary and force upper case for sequence
                (x == base) for x in list(sequence[trim_start:(None if trim_end == 0 else -trim_end)].upper())  # force upper case for bases too
            ] for base in matrix_bases.upper()
        ],dtype=np.int8
    )

    # If any characters weren't matched (which you can find because all elements
    # at that position are 0), make them into Ns.
    # (Could teach it to accept other IUPAC characters here...)

    # this doesnt work because binarised bases are int8s, we are expressing
    # non genomic characters as absence of A,C,G and T
    #binarised_bases[ np.all(binarised_bases == 0, axis=1),:] = 0.25

    # if needed, add padding of Ns on either side of sequence and return
    if paddingL > 0 or paddingR>0:

        return padding_dna(binarised_bases, paddingL, paddingR)
    # otherwise just return the binarised bases
    return binarised_bases


def load_sequence(filename, start=None, end=None):
    """
    Takes a file containing DNA sequence information and returns a matrix
    representation of the sequence.

    Parameters
    ----------
    filename : str
        The name of the file to be opened, including its path.
    start : int, optional
        Start position within the sequence. 1-based. Default None (whole
        sequence).
    end : int, optional
        End position within the sequence. 1-based. Default None (whole
        sequence).
    """
    with open(filename, 'r') as sequence_file:
        # read in file, skipping the first > comment line and
        # concatenate the rest, removing line breaks
        sequence = ''.join(sequence_file.readlines()[1:]). \
            replace('\n', '')
        # If there's a start and end specified, use those coordinates
        # to trim the sequence down...
        # (Use 1-based indexing like most genomic things do...)
        if start and end:
            sequence = sequence[(start - 1):(end - 1)]
            # if using other filetypes this BioPython command might come in
            # useful again:
            # for seq_record in SeqIO.parse(sequence_filename, sequence_filetype):
            # for sequence_filename in os.listdir(sequences_dir):
    # turn the sequence into a binarised matrix
    # use seq_record.seq if using BioPython
    return bases_to_binary(sequence)


def padding_dna(binarised_bases, paddingL, paddingR):
    """
    Takes DNA sequence(s) as an array and pads with Ns on both ends.

    Parameters
    ----------
    binarised_bases : array_like
        Matrix representation of DNA sequence(s).
    paddingL : int
        Number of Ns (ie 4 0.25s) to add to the start of the sequence.
    paddingR : int
        Number of Ns (ie 4 0.25s) to add to the end of the sequence.
    """

    # If it's 2D, ie a single DNA sequence
    if binarised_bases.ndim == 2:
        padsL = 0.25 * np.ones([binarised_bases.shape[0], paddingL])
        padsR = 0.25 * np.ones([binarised_bases.shape[0], paddingR])
        return np.append(padsL, np.append(binarised_bases, padsR, axis=1), axis=1)
    # If it's 3D, ie a series of many sequences
    elif binarised_bases.ndim == 3:
        padsL = 0.25 * np.ones([binarised_bases.shape[0], paddingL, binarised_bases.shape[-1]])
        padsR = 0.25 * np.ones([binarised_bases.shape[0], paddingR, binarised_bases.shape[-1]])
        return np.append(padsL, np.append(binarised_bases, padsR, axis=1), axis=1)
    # Otherwise, this function can't do it
    else:
        raise ValueError('Invalid number of dimensions for DNA matrix (' + str(binarised_bases.ndim) + ')')


def read_psm(filename, psm_format='compete'):
    """

    Parameters
    ----------
    filename : str
        The name of the PSM file to read.
    psm_format : str
        The format of the PSM file.
            compete: wide-format 4xL matrix of floats

    Returns
    -------
    An array representation of the PPM.

    """
    if psm_format == 'compete':
        ppm = pandas.read_csv(filename, sep='\t', skiprows=1, header=None)
        # Remove the first column, which is just the letters A, C, G and T
        ppm = ppm.ix[:, 1:]
        # Turn into a NumPy array (columnwise as in the file)
        ppm = ppm.as_matrix()
        return ppm
    else:
        raise ValueError('Unrecognised PSM format ' + psm_format)


#####################################
# parameter sampling
################################

def log_uniform(a, b):
    interim = np.random.uniform(0, 1)
    return_value = 10 ** ((np.log10(b) - np.log10(a)) * interim + np.log10(a))
    return return_value


def sqrt_uniform(a, b):
    interim = np.random.uniform(0, 1)
    return_value = (b - a) * np.sqrt(interim) + a
    return return_value


# helper function to split up training data
def sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]


# If this is called as a script, perform unit testing based on examples in
# each function's docstring.
if __name__ == "__main__":
    import doctest

    doctest.testmod()



def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def using_tile_and_stride(input_array, N):
    # pad to right square shape
    input_array = np.pad(input_array, ((0, 0), (0, N - 1)), mode='constant')
    arr = np.tile(input_array, (1, 1))
    row_stride, col_stride = arr.strides
    arr.strides = row_stride - col_stride, col_stride
    return arr


def get_shape(array, axis =-1):
    return(np.shape(array)[axis])

def pad_last_axis(array,len, value =np.int8(0)):
    return(np.pad(np.expand_dims(array, axis =0), ((0,0),(0,0),(0,len - array.shape[-1])), 'constant', constant_values=(value, value)))



def make_list(arr):
    if arr is not None:
        return arr.tolist()
    else:
        return None




# pearson correlation coefficient for tensors
def Kp_cor(x, y):
    # pearson correlation between tensors
    # x and y should have same length.
    # x = K.variable(x)
    # y = K.variable(y)
    mx = K.mean(x, axis=-1, keepdims=True)
    my = K.mean(y, axis=-1, keepdims=True)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym, axis=-1)
    # ss squares each element and returns sum
    r_den = K.sqrt(K.sum(xm * xm, axis=-1) * K.sum(ym * ym, axis=-1))
    r_den = K.clip(r_den, K.epsilon(), np.infty)
    r = r_num / r_den
    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    return K.clip(r, -1.0, 1.0)


def mean_classification_pearson_correlation(true, pred):
    return (K.mean(Kp_cor(K.cast(K.argmax(true), K.floatx()), K.cast(K.argmax(pred), K.floatx()))))


def p_cor(x, y):
    # pearson correlation between tensors
    # x and y should have same length.
    # x = K.variable(x)
    # y = K.variable(y)
    mx = np.mean(x, axis=-1, keepdims=True)
    my = np.mean(y, axis=-1, keepdims=True)
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym, axis=-1)
    # ss squares each element and returns sum
    r_den = np.sqrt(np.sum(xm * xm, axis=-1) * np.sum(ym * ym, axis=-1))
    # making sure we don't divide by zero (those sequences that are only genomic background?!)
    r = r_num / np.max(np.vstack((K.epsilon()*np.ones_like(r_den), r_den)), axis=0)
    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    return np.clip(r, -1.0, 1.0)


def compute_receptive_field( dilation_depth,dilation_kernel_size, nb_stacks):
    receptive_field = nb_stacks * (2 ** dilation_depth * dilation_kernel_size) - (nb_stacks - 1)
    #receptive_field_ms = (receptive_field * 1000) / desired_sample_rate
    return receptive_field#, receptive_field_ms

def skip_out_of_receptive_field(func):
    receptive_field= compute_receptive_field()

    def wrapper(y_true, y_pred):
        y_true = y_true[:, receptive_field - 1:, :]
        y_pred = y_pred[:, receptive_field - 1:, :]
        return func(y_true, y_pred)

    wrapper.__name__ = func.__name__

    return wrapper

def print_t(tensor, label):
    tensor.name = label
    tensor = K.printing.Print(tensor.name, attrs=('__str__', 'shape'))(tensor)
    return tensor

def make_soft(y_true, fragment_length, nb_output_bins, train_with_soft_target_stdev, with_prints=False):
    receptive_field, _ = compute_receptive_field()
    n_outputs = fragment_length - receptive_field + 1

    # Make a gaussian kernel.
    kernel_v = scipy.signal.gaussian(9, std=train_with_soft_target_stdev)
    print(kernel_v)
    kernel_v = np.reshape(kernel_v, [1, 1, -1, 1])
    kernel = K.variable(kernel_v)

    if with_prints:
        y_true = print_t(y_true, 'y_true initial')

    # y_true: [batch, timesteps, input_dim]
    y_true = K.reshape(y_true, (-1, 1, nb_output_bins, 1))  # Same filter for all output; combine with batch.
    # y_true: [batch*timesteps, n_channels=1, input_dim, dummy]
    y_true = K.conv2d(y_true, kernel, border_mode='same')
    y_true = K.reshape(y_true, (-1, n_outputs, nb_output_bins))  # Same filter for all output; combine with batch.
    # y_true: [batch, timesteps, input_dim]
    y_true /= K.sum(y_true, axis=-1, keepdims=True)

    if with_prints:
        y_true = print_t(y_true, 'y_true after')
    return y_true


def make_targets_soft(func):
    """Turns one-hot into gaussian distributed."""

    def wrapper(y_true, y_pred):
        y_true = make_soft(y_true)
        y_pred = y_pred
        return func(y_true, y_pred)

    wrapper.__name__ = func.__name__

    return wrapper




def categorical_mean_squared_error(y_true, y_pred):
    """MSE for categorical variables."""
    return K.mean(K.square(K.argmax(y_true, axis=-1) -
                           K.argmax(y_pred, axis=-1)))



def all_equal(iterator):
  try:
     iterator = iter(iterator)
     first = next(iterator)
     return all(np.array_equal(first, rest) for rest in iterator)
  except StopIteration:
     return True

def vector_mse(y_true, y_pred):
    diff2 = (y_true - y_pred)**2
    return K.mean(K.sum(diff2, axis = -1))


# custom loss using the pearson correclation is taken from zhong
def custom_loss(y_true, y_pred, method = 'pearson'):
    if method in ['pearson']:
        return -K.log(K.sqrt(1+p_cor(y_true,y_pred))/K.sqrt(2.0))
    elif method == 'mse':
        return vector_mse(y_true, y_pred)


def measure_h(list_true,list_pred):
    h = np.zeros(shape = list_true[0].shape[0])
    for i in range(len(list_true)):
        h=h-custom_loss(K.variable(list_true[i]),K.variable(list_pred[i]))
    h = K.exp(h)
    return h.eval()


def get_session(gpu_fraction=0.1):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options = gpu_options, intra_op_parallelism_threads = num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    ktf.set_session(get_session())

# Define pearson correlation coefficient
# REF: https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
def cor_pcc(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

# Compute R squared - KERAS metrics
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# Compute matthews correlation coef
def mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())




def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)




def pandas_df_reindex(df, idx, method):
    """ Reindexes the given dataframe with index = idx, and per column methods as given in the map 'method', which
    must be a map from column names in df to either a tuple ("fill_value", value_to_fill) which will fill the column
    with value_to_fill, or to one of the methods accepted by pandas.Series.reindex:
    {None, 'backfill'/'bfill', 'pad'/'ffill', 'nearest'}."
    """
    if isinstance(method, str) or method is None:
        return df.reindex(idx, method=method)
    if not isinstance(method, dict):
        raise ValueError("Method must be dict, string, or None")

    #  Reindex each column by themselves and remember the result in reindexed_columns
    reindexed_columns = dict()
    for column_name in method:
        column_method = method[column_name]
        if isinstance(column_method, tuple):
            (_, fill_value) = column_method
            reindexed_columns[column_name] = df[column_name].reindex(idx, fill_value=fill_value)
        else:
            reindexed_columns[column_name] = df[column_name].reindex(idx, method=column_method)

    df = df.reindex(idx)
    #  Add the reindexed columns back in
    for column_name in reindexed_columns:
        df[column_name] = reindexed_columns[column_name]
    return df
