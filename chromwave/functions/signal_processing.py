
from __future__ import division




import numpy as np
import scipy.io.wavfile
import scipy.signal

from keras.utils.np_utils import to_categorical
import pandas
from sklearn.model_selection import GridSearchCV

from scipy.ndimage.filters import gaussian_filter1d
from copy import copy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error as mse

from .utils import pandas_df_reindex

def float_to_int(x, u =1):
    # scale if necessary
    x *= u
    x = np.floor(x)
    return x.astype('int')


def int_to_float(x, u =1):
    # scale if necessary
    x =x / np.float(u)
    return x




def float_to_uint8(x, u =255):
    x += 1.
    x /= 2.
    uint8_max_value = np.int(u)
    x *= uint8_max_value
    x = x.astype('uint8')
    return x




def categorize(n_bins):
    def wrapper(arr):
        return(to_categorical(arr, n_bins))
    return wrapper



def missing_elements(L, start = None, end= None):
    if isinstance(L,list):
        L = np.array(L)
    if start is None:
        start = L[0]
    if end is None:
        end =  L[-1]
    L = L[L <= end]
    return sorted(set(range(int(start), int(end) + 1)).difference(L))




def add_missing_elements(pandas_data_frame, seqnames_column_name='seqnames', start_column_name = 'start',
                         end_column_name = None, score_column_idx = 2, score_column_name=None, value = None,
                         start = None, end=None):
    # getting rid of redundant columns
    if score_column_name is None:
        score_column_name=pandas_data_frame.columns[score_column_idx]

    # if the start and end columns are given and the ranges are not 1bp, we need to unroll those ranges
    if end_column_name is not None:
        pandas_data_frame = pandas_data_frame[
            [seqnames_column_name, start_column_name, end_column_name, score_column_name]]
        pandas_data_frame.dropna(axis=0, inplace=True)
        non_zero_ranges = pandas_data_frame.loc[
                          pandas_data_frame[end_column_name] - pandas_data_frame[start_column_name] > 1, :]
        if len(non_zero_ranges)>0:
            print('The file provided has missing values - these will be replaced by the provided value or the genomic mean.'
                  'If you are using bed files please ensure that the ranges in provided bedfiles are of size 1, ' \
                                       'e.g. there is a score for each basepair position')

    pandas_data_frame = pandas_data_frame[[seqnames_column_name, start_column_name, score_column_name]]
    pandas_data_frame.dropna(axis=0, inplace=True)

    if end is not None:
        pandas_data_frame = pandas_data_frame[pandas_data_frame[start_column_name] <= end]
    if start is not None:
        pandas_data_frame = pandas_data_frame[pandas_data_frame[start_column_name] >= start]

    m=missing_elements(np.array(pandas_data_frame[start_column_name]).astype(int), start=start, end = end)
    if value is None:
        av=np.mean(pandas_data_frame[score_column_name])
    else:
        av=value
    m = pandas.DataFrame(m, columns=[start_column_name])
    m[seqnames_column_name]=pandas_data_frame[seqnames_column_name].unique()[0]

    m[score_column_name]=av
    # cols of m have to be same order as in pandas_data_frame
    m=m[[seqnames_column_name,start_column_name,score_column_name]]
    tmp = pandas_data_frame.append(m)
    tmp= tmp.sort_values(start_column_name)
    tmp.set_index(start_column_name, inplace=True)

    return tmp


def shift_positive(seq):
    if  np.any(seq<0):
        seq=seq- np.min(seq)
    return seq



def savgol_filter(arr,window_length = 151, polyorder = 2):
    if window_length is not None and polyorder is not None:
        return scipy.signal.savgol_filter(arr, window_length, polyorder)
    else:
        return arr


def runs_of_ones_array(bits, thresh):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    run_len = run_ends - run_starts
    idx = [range(run_starts[i], run_ends[i]) for i in range(0, run_len.shape[0]) if run_len[i] < thresh]
    idx = [item + 1 for sublist in idx for item in sublist]
    bounded[idx] = 0
    return bounded[1:bounded.shape[-1] - 1]


def savitzky_golay_piecewise(x_thresh,run_thresh,data, window_length, polyorder):
    x = np.ones(shape = data.shape)
    x[data<=x_thresh] = 0
    xvals=runs_of_ones_array(x, run_thresh)
    idx = xvals> 0
    y_smooth =  data
    y_smooth[idx]=savgol_filter( data[idx],window_length=window_length,polyorder=polyorder)
    y_smooth[xvals== 0]= np.mean(data[xvals== 0])
    return y_smooth

def return_signal_pieces(data,x_thresh,run_thresh):
    x = np.ones(shape = data.shape)
    x[data<=x_thresh] = 0
    xvals=runs_of_ones_array(x, run_thresh)
    idx = xvals> 0
    return data[idx]


class SavgolSmoother(BaseEstimator, ClassifierMixin):


    def __init__(self, polyorder=2, window_length = 51,lam=0.5):
        """
        Called when initializing the classifier
        """
        self.polyorder =  polyorder
        self.window_length = window_length
        self.lam=lam

    def get_smooth_signal(self,seq,y=None):

        return(np.array(savgol_filter(seq, window_length=self.window_length, polyorder=self.polyorder)))

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        assert (type(self.polyorder) == int), "polyorder parameter must be integer"
        assert (type(self.window_length) == int), "window_length parameter must be string"

        if self.window_length>self.polyorder:
            self.smooth_Signal_ = [self.get_smooth_signal(seq) for seq in X]
        else:
            self.smooth_Signal_=None
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "smooth_Signal_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        if self.smooth_Signal_ is not None:
            return([self.get_smooth_signal(seq) for seq in X])
        else:
            return None

    def score(self, X, y=None):
        # counts number of values bigger than mean
        seq_pred= self.predict(X)
        if seq_pred is not None:
            return (-(np.mean([mse(seq, seq_p) + self.lam * mse(seq_p[:-1], seq_p[1:]) for (seq, seq_p) in zip(X, seq_pred)])))
        else:
            return -np.nan



class GaussianFilterSmoother(BaseEstimator, ClassifierMixin):

    def __init__(self, sigma=10,lam=0.5, truncate=4.0):
        """
        Called when initializing the classifier
        """
        self.sigma =  sigma
        self.lam=lam
        self.truncate=truncate
    def get_smooth_signal(self,seq,y=None):

        return(np.array(gaussian_filter1d(copy(seq), sigma=self.sigma, truncate=self.truncate)))

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self.smooth_Signal_ = [self.get_smooth_signal(seq) for seq in X]

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "smooth_Signal_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        if self.smooth_Signal_ is not None:
            return([self.get_smooth_signal(seq) for seq in X])
        else:
            return None

    def score(self, X, y=None):
        # counts number of values bigger than mean
        seq_pred= self.predict(X)
        if seq_pred is not None:
            return (-(np.mean([mse(seq, seq_p) + self.lam * mse(seq_p[:-1], seq_p[1:]) for (seq, seq_p) in zip(X, seq_pred)])))
        else:
            return -np.nan




def gridsearchCV_preprocessing_params(y, smoother='savgol',param_grid = {"polyorder": range(4), "window_length": range(1,52,2)}):
    if smoother=='savgol':
        smooth_gridsearch = GridSearchCV(SavgolSmoother(),param_grid,cv=None,verbose=1)
    elif smoother=='gaussian_1d':
        smooth_gridsearch =GridSearchCV(GaussianFilterSmoother(), param_grid, cv=None, verbose=1)
    else:
        print('Error, parameter optimization for chosen smoothing algorithm not implemented, please provide hyperparameters.')
        return
    smooth_gridsearch.fit(y)
    return smooth_gridsearch.best_params_,smooth_gridsearch.best_estimator_.smooth_Signal_


