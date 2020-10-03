import glob
from Bio import SeqIO
from copy import copy
from joblib import Parallel, delayed
import keras.backend as K
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot
import multiprocessing
import numpy
import os
import pandas
import roman
from scipy.ndimage.filters import gaussian_filter1d
from skimage.util.shape import view_as_windows, view_as_blocks
from sklearn import utils
from sklearn.model_selection import train_test_split
import time

from .functions import utils
from .functions.signal_processing import gridsearchCV_preprocessing_params, missing_elements
from .functions.signal_processing import int_to_float
from .functions.signal_processing import shift_positive, add_missing_elements, savgol_filter, float_to_int
from .vis import plot

_EPSILON = K.epsilon()


class RuntimeDataset:
    """
    Core data holder for the biological neural network process. Ensures valid data is present for the current runtime, and unpacks the data in the most relevant manner.
    Training/Test/Val data split only if in training mode - this is handled through the file system by passing a test and val fraction.
    If not in trianing mode, genomic data and binding profile data will be loaded and correctly processed as determined
    in the preprocessing parameter set. Additional

    # Properties
        verbose
        parallelize
        n_output_features
        _val_fraction
        _remove_unmapped_training_regions
        _chr_vector
        preprocessing_params
        _is_validated
        _is_training_loaded
        _binding_profile_data
        save_data
        _pad_to_center
        step_size
        _train_test_split_by_chr
        load_existing_data
        fragment_length
        _using_full_genomic_sequences
        zero_based
        class_weight_cap
        _runtime_data_source
        _runtime_processed_data_source
        _runtime_output
        _shuffle_sequences
        data_format
        _include_rc
        _is_training
        _is_profiling
        _set_seed
        plot_fit_diganostics
        _test_fraction


    # Methods

        _compute_GC_content
        _compute_class_weights
        _compute_reverse_complements
        _discretize_binding_profile_data
        _evaluate_data_type
        _load_binding_profile_data
        _load_data
        _load_existing_data
        _load_genome_data
        _load_genome_data_timed
        _load_profile
        _load_profile_data
        _load_training
        _load_training_timed
        _pad_DNA
        _process_profile
        _set_training_mode
        _smooth_discrete_data
        _smooth_profile_data
        _split_binding_profile_data
        _validate_preprocessing_params
        get_methods
        invert_discretizing
        is_training
        is_training_loaded
        is_validated
        load_data
        pre_process
        validate


    """

    # Training runtime

    # Short-circuits
    _is_validated = False  # Ensue we do not repeat validation steps when unnecessary
    _is_training = False
    _is_training_loaded = False  # Ensures we don't load multiple times

    def __init__(self, filesystem):

        # Applying some defaults here

        # input and output directories
        self._runtime_data_source = filesystem.get_genome_directory()  # genomic data directory
        self._runtime_output = os.path.join(filesystem.get_output_directory(), "data")
        self._runtime_processed_data_source = self._runtime_output  # data gets saved in this directory

        self.verbose = 0
        # Number of binding profiles
        self.n_output_features = 1

        # saved data loading/saving
        self.load_existing_data = False
        self.save_data = False

        # Data processing & loading
        self.preprocessing_params = None
        self.data_format = 'raw_counts'  # assuming raw counts
        self.plot_fit_diganostics = False
        self.zero_based = False  # whether the profiles are in a zero based based format
        self._using_full_genomic_sequences = True  # needed later on if full genomic sequences are read in (ie start and end info in bed files of profile range from 0 to len(chrom) or if start, end are eg promoter coordinates
        self._chr_vector = None  # pass names of the sequences when all sequences are read from one file, e.g. if passing promoter sequences. Needed for train/test/splitting
        self.parallelize = False  # should several CPUs be used in parallel, useful if many sequences e.g. human promoter data
        self._pad_to_center = False  # when sequences are padded, should they be padded both from left and right so the sequence is centered?

        self.fragment_length = 2000  # size of windows
        self.step_size = None  # step size of windows

        # Parameters for training
        self.class_weight_cap = [40]
        # splitting into test/val sets and shuffling
        self._set_seed = None
        self._shuffle_sequences = False
        self._train_test_split_by_chr = True  # split sequences into train test random or split chromosomes in train test? Recommended to avoid information leakage
        self._include_rc = True  # is reverse complement included
        self._remove_unmapped_training_regions = None
        self._binding_profile_data = filesystem.get_profile_directory()

        # we are in training mode if test and val fractions were given to the filesystem, in this case we split the
        # data and include reverse complement and compute class weights.
        if filesystem.is_training():
            self._set_training_mode(filesystem.get_validation_fraction(), filesystem.get_test_fraction())

        self._is_profiling = False
        if filesystem.is_profiling():
            self._is_profiling = True

        self._is_training_loaded = False
        self._is_validated = False

    def _set_training_mode(self, val_fraction=0.2, test_fraction=0.1):
        self._is_training = True
        # Ensure that we can generate an evaluation after the fact
        self._test_fraction = test_fraction
        self._val_fraction = val_fraction

    def validate(self):
        if self._is_validated:
            return True

        # Lets check the output
        if not os.path.isdir(self._runtime_output):
            os.makedirs(self._runtime_output)

        # Does the file actually exist :P
        if not os.path.isfile(self._runtime_data_source) and not os.path.isdir(self._runtime_data_source):
            print('runtime_data_source does not exist')
            return False

        # Evaluate whether or not it is a known file type
        if not self._evaluate_data_type():
            print('Data type not recognised.')
            return False

        # Evaluate testing variables
        if self._is_training:
            if not self._test_fraction or not self._val_fraction:
                print('Cannot be in training mode if no test and val fractions passed.')
                return False

        # do we have all preprocessing parameters we need?
        if not self._validate_preprocessing_params():
            return False

        self._is_validated = True
        return True

    def is_validated(self):
        return self._is_validated

    def is_training(self):
        return self._is_training

    def is_genome_data_loaded(self):
        return self._is_genome_data_loaded

    def are_profiles_loaded(self):
        return self._are_profiles_loaded

    def is_training_loaded(self):
        return self._is_training_loaded

    def _validate_preprocessing_params(self):

        if not self.data_format in ['raw_counts', 'processed_counts']:
            print('Data formate not recognised')
            return False

        if self.preprocessing_params is None:
            print('No preprocessing parameters given.')
            return False
        else:
            for params in self.preprocessing_params:
                # Need to for streching to discretise
                if not "u" in params:
                    print('Preprocessing parameter "u" missing')
                    return False
                if not "smooth_signal" in params:
                    print('Preprocessing parameter "smooth_signal" missing')
                    return False
                if not "normalise_read_counts" in params:
                    print('Preprocessing parameter "normalise_read_counts" missing')
                    return False
                if params['smooth_signal']:
                    if not "smoothing_function" in params:
                        print('Preprocessing parameter "smoothing_function" missing')
                        return False
                    if not params['smoothing_function'] in ['savgol_filter', 'gaussian_filter1d']:
                        print('Smoothing function not implemented, choose from gaussian_filter1d and savgol_filter')
                        return False
                if self.data_format == 'raw_counts':
                    if not "times_median_coverage_max" in params:
                        print('Preprocessing parameter "times_median_coverage_max" missing')
                        return False

        return True

    def _evaluate_data_type(self):
        # runtime data source needs to be a directory with fasta files or a path to a fasta file
        if os.path.isfile(self._runtime_data_source):
            filename, extension = os.path.splitext(self._runtime_data_source)
            extension = extension.lower()
            if not extension in [".fa", ".fasta", ".fsa"]:
                print('No fasta files found in runtime_data_source directory.')
                return False

        elif os.path.isdir(self._runtime_data_source):
            sequences = glob.glob(os.path.join(self._runtime_data_source, "*.fa"))
            sequences.extend(glob.glob(os.path.join(self._runtime_data_source, "*.fasta")))
            sequences.extend(glob.glob(os.path.join(self._runtime_data_source, "*.fsa")))
            if not len(sequences) > 0:
                print('No fasta files found in runtime_data_source directory.')
                return False
        else:
            return False

        # binding profiles need to be

        if isinstance(self._binding_profile_data, list):
            for binding_profile_data_source in self._binding_profile_data:
                if os.path.isfile(binding_profile_data_source):
                    filename, extension = os.path.splitext(binding_profile_data_source)
                    if not extension in ['.csv', '.bed', '.mat']:
                        print('Binding data can only have the following file formats: csv, bed, mat.')
                        return False
                else:
                    print('Binding data not found.')
                    return False
        elif isinstance(self._binding_profile_data, str) and os.path.isfile(self._binding_profile_data):
            filename, extension = os.path.splitext(self._binding_profile_data)
            if not extension in ['.csv', '.bed', '.mat']:
                print('Binding data can only have the following file formats: csv, bed, mat.')
                return False
        else:
            print('Binding profile data format not recognised, please pass files or list of files of the correct format')
            return False

        return True

    def load_data(self):

        if not self.validate():
            raise Exception("Your runtime dataset was not in a valid state.")

        if self._is_profiling:
            time_start = time.clock()
            self.verbose = 1
            self._load_data_timed()
        else:
            self._load_data()

        if self._is_training:
            if self._is_profiling:
                self.verbose = 1
                self._load_training_timed()
            else:
                self._load_training()

        if self._is_profiling:
            print("All data loaded in: " + str(time.clock() - time_start) + " seconds")

    def _load_training_timed(self):
        time_start = time.clock()
        self._load_training()
        print("Training data loaded in: " + str(time.clock() - time_start) + " seconds")

    def _load_training(self):
        print("Beginning training data load")

        if not self._is_training:
            raise AssertionError("Cannot load training data whilst not in training mode.")

        if self._is_training_loaded:
            return

        if not self.validate():
            raise Exception("Your runtime dataset was not in a valid state.")

        if not self.is_genome_data_loaded():
            raise Exception('Genome data was not loaded.')

        if not self.are_profiles_loaded():
            raise Exception('Binding profile data was not loaded.')

        # Handle loading the training data set

        if self.x_data is None:
            raise Exception('Genome data has not been loaded properly')

        if self.y_data is None:
            raise Exception('Binding data has not been loaded properly')

        x_data = self.x_data
        y_data_smooth = self.y_data
        y_data_orig = self.profile_data_split
        y_data = self.y_data_discr

        if isinstance(self._chr_vector, str):
            if self.load_existing_data:
                if os.path.exists(os.path.join(self._runtime_processed_data_source, 'chr_vector.json')):
                    self._chr_vector = utils.load_json(
                        os.path.join(self._runtime_processed_data_source, 'chr_vector.json'))
                else:
                    print('file ' + os.path.join(self._runtime_processed_data_source,
                                                 'chr_vector.json') + ' does not exist. Proceeding without chr info for train-test-split. ')
                    self._chr_vector = None
            else:
                if os.path.exists(self._chr_vector):
                    seqnames_stratify = pandas.read_csv(self._chr_vector, sep='\t', header=None)
                    seqnames_stratify[3] = seqnames_stratify[3].astype(str)
                    seq_stratify = [seqnames_stratify[seqnames_stratify[3] == a].iloc[0][0] for a in self.chr_info]
                    self._chr_vector = seq_stratify
                    if self.save_data:
                        # saving because chr info is alwyas same as in fasta file
                        utils.save_json(seq_stratify,
                                        os.path.join(self._runtime_processed_data_source, 'chr_vector.json'))
                else:
                    print(
                        'file ' + self._chr_vector + ' does not exist. Proceeding without chr info for train-test-split. ')
                    self._chr_vector = None

        if self._remove_unmapped_training_regions is not None:
            # we are removing training sequences that that have too many constant values
            def has_too_much_background(arr, frac):
                if frac is not None:
                    classes, counts = numpy.unique(arr, return_counts=True)
                    return numpy.any(counts > len(arr) * frac)
                else:
                    return False

            if not isinstance(self._remove_unmapped_training_regions, list):
                if isinstance(self._remove_unmapped_training_regions, float):
                    self._remove_unmapped_training_regions = [self._remove_unmapped_training_regions] * len(y_data)

            # test for each sequences (and each profile type) if it has too many constant values
            tmp = [[numpy.apply_along_axis(has_too_much_background, 1, y0, frac=f) for y0 in y] for (y, f) in
                   zip(y_data, self._remove_unmapped_training_regions)]
            # get a bool mask
            masks = [numpy.all(numpy.vstack([tmp[i][j] for i in range(len(tmp))]), axis=0) for j in range(len(self.chr_info))]

            if self._using_full_genomic_sequences:
                # some safeguarding we are not removeing all sequences in the dataset
                if numpy.array([m.all() for m in masks]).any():
                    print('Error: attempted to remove all sequences on whole chromosome. Signal too sparse for chosen _remove_unmapped_training_regions parameter. Try running with a lower fraction.')
                elif numpy.array([m.all() for m in masks]).all():
                    print(
                        'Error: attempted to remove all sequences on whole chromosome. Signal too sparse for chosen _remove_unmapped_training_regions parameter. Try running with a lower fraction.')
                else:
                    print('Removing ' + str(numpy.sum(numpy.hstack(masks))) + ' fragments that have have too many constant values in all of the provided profiles.')

                # removing sequences
                y_data = [[y[~m] for (y, m) in zip(y_data_i, masks)] for y_data_i in y_data]
                y_data_smooth = [[y[~m] for (y, m) in zip(y_data_smooth_i, masks)] for y_data_smooth_i in y_data_smooth]
                y_data_orig = [[y[~m] for (y, m) in zip(y_data_orig_i, masks)] for y_data_orig_i in y_data_orig]
                x_data = [x[~m] for (x, m) in zip(x_data, masks)]

            if not self._using_full_genomic_sequences:
                # if not ful chrs are given the numpy arrays in y_data had all shape (1,length) and are now empty arrays that need to be removed, similarly all other sequence data that is associated with them.
                # index to remove
                idx = [[i for i in range(0, len(y_data_i)) if 0 in y_data_i[i].shape] for y_data_i in y_data]
                idx = numpy.unique([i for i in idx])
                if len(idx) > 0:
                    print('Removing ' + str(len(idx)) + ' fragments that are empty.')
                    self.chr_info = [self.chr_info[i] for i in range(0, len(self.chr_info)) if i not in idx]
                    y_data = [[y_data_i[i] for i in range(0, len(y_data_i)) if i not in idx] for y_data_i in y_data]
                    y_data_smooth = [[y_data_i[i] for i in range(0, len(y_data_i)) if i not in idx] for y_data_i in y_data_smooth]
                    y_data_orig = [[y_data_i[i] for i in range(0, len(y_data_i)) if i not in idx] for y_data_i in y_data_orig]
                    x_data = [x_data[i] for i in range(0, len(x_data)) if i not in idx]
                    self._chr_vector = [self._chr_vector[i] for i in range(0, len(self._chr_vector)) if i not in idx]

        def classes_and_counts(arr):
            return numpy.unique(arr.flatten(), return_counts=True)

        # counting how often each class occurs in the dataset
        _class_counts = [classes_and_counts(numpy.vstack(y)) for y in y_data]

        # _compute_class_weights needs the fragment length, this is the same as .y_data_discr[0][0].shape[-1]
        self._compute_class_weights(_class_counts, y_data[0][0].shape[-1])

        def categorize(arr, _output_bins):
            return to_categorical(arr, _output_bins)

        print("Subsetting training data with test proportion: " + str(
            self._test_fraction) + ' and validation proportion ' + str(self._val_fraction))

        # if we want to split sequences according to the chromosomes into train/test/val rather than just random
        # recommended to avoid information leakage
        if self._train_test_split_by_chr:
            if self._chr_vector is None:
                train_chr, test_chr = train_test_split(self.chr_info, test_size=self._test_fraction,
                                                       random_state=self._set_seed, shuffle=self._shuffle_sequences,
                                                       stratify=None)
                train_chr, val_chr = train_test_split(train_chr, test_size=self._val_fraction,
                                                      random_state=self._set_seed, shuffle=self._shuffle_sequences,
                                                      stratify=None)
                self.train_chr_info = train_chr
                self.val_chr_info = val_chr
                self.test_chr_info = test_chr
            else:

                chrs = numpy.unique(self._chr_vector)
                train_chr, test_chr = train_test_split(chrs, test_size=self._test_fraction,
                                                       random_state=self._set_seed, shuffle=self._shuffle_sequences,
                                                       stratify=None)
                train_chr, val_chr = train_test_split(train_chr, test_size=self._val_fraction,
                                                      random_state=self._set_seed, shuffle=self._shuffle_sequences,
                                                      stratify=None)

                self.train_chr_info = numpy.array(self.chr_info)[numpy.isin(numpy.array(self._chr_vector), train_chr)].tolist()
                self.val_chr_info = numpy.array(self.chr_info)[numpy.isin(numpy.array(self._chr_vector), val_chr)].tolist()
                self.test_chr_info = numpy.array(self.chr_info)[numpy.isin(numpy.array(self._chr_vector), test_chr)].tolist()

            y_train_smooth = [numpy.vstack([y[self.chr_info.index(c)] for c in self.train_chr_info]) for y in y_data_smooth]
            y_test_smooth = [numpy.vstack([y[self.chr_info.index(c)] for c in self.test_chr_info]) for y in y_data_smooth]
            y_val_smooth = [numpy.vstack([y[self.chr_info.index(c)] for c in self.val_chr_info]) for y in y_data_smooth]

            y_train_orig = [numpy.vstack([y[self.chr_info.index(c)] for c in self.train_chr_info]) for y in y_data_orig]
            y_test_orig = [numpy.vstack([y[self.chr_info.index(c)] for c in self.test_chr_info]) for y in y_data_orig]
            y_val_orig = [numpy.vstack([y[self.chr_info.index(c)] for c in self.val_chr_info]) for y in y_data_orig]

            y_train_discr = [numpy.vstack([y[self.chr_info.index(c)] for c in self.train_chr_info]) for y in y_data]
            y_test_discr = [numpy.vstack([y[self.chr_info.index(c)] for c in self.test_chr_info]) for y in y_data]
            y_val_discr = [numpy.vstack([y[self.chr_info.index(c)] for c in self.val_chr_info]) for y in y_data]

            x_train = numpy.vstack([x_data[self.chr_info.index(c)] for c in self.train_chr_info])
            x_test = numpy.vstack([x_data[self.chr_info.index(c)] for c in self.test_chr_info])
            x_val = numpy.vstack([x_data[self.chr_info.index(c)] for c in self.val_chr_info])

            y_categorical_labels = [[numpy.array(
                numpy.apply_along_axis(categorize, axis=1, arr=y, _output_bins=_output_bins), dtype='int8') for y
                in y_dat] for (y_dat, _output_bins) in zip(y_data, self._output_bins)]

            y_train = [numpy.vstack([y[self.chr_info.index(c)] for c in self.train_chr_info]) for y in
                       y_categorical_labels]
            y_test = [numpy.vstack([y[self.chr_info.index(c)] for c in self.test_chr_info]) for y in
                      y_categorical_labels]
            y_val = [numpy.vstack([y[self.chr_info.index(c)] for c in self.val_chr_info]) for y in
                     y_categorical_labels]

        else:
            y_data_smooth_tmp = numpy.stack([numpy.vstack(y) for y in y_data_smooth], axis=2)
            y_data_orig_tmp = numpy.stack([numpy.vstack(y) for y in y_data_orig], axis=2)
            y_data_tmp = numpy.stack([numpy.vstack(y) for y in y_data], axis=2)
            x_data_tmp = numpy.vstack(x_data)
            x_data_tmp, x_test, y_data_tmp, y_test_discr, y_data_smooth_tmp, y_test_smooth, y_data_orig_tmp, y_test_orig = \
                train_test_split(x_data_tmp, y_data_tmp, y_data_orig_tmp, y_data_smooth_tmp, test_size=self._test_fraction,
                                 random_state=self._set_seed, shuffle=self._shuffle_sequences)
            x_train, x_val, y_train_discr, y_val_discr, y_train_smooth, y_val_smooth, y_train_orig, y_val_orig = \
                train_test_split(x_data_tmp, y_data_tmp, y_data_orig_tmp, y_data_smooth_tmp, test_size=self._val_fraction,
                                 random_state=self._set_seed, shuffle=self._shuffle_sequences)
            # turngin all y-data back into lists
            y_train_smooth = [y_train_smooth[:, :, i] for i in range(0, y_train_smooth.shape[2])]
            y_test_smooth = [y_test_smooth[:, :, i] for i in range(0, y_test_smooth.shape[2])]
            y_val_smooth = [y_val_smooth[:, :, i] for i in range(0, y_val_smooth.shape[2])]

            y_train_discr = [y_train_discr[:, :, i] for i in range(0, y_train_discr.shape[2])]
            y_test_discr = [y_test_discr[:, :, i] for i in range(0, y_test_discr.shape[2])]
            y_val_discr = [y_val_discr[:, :, i] for i in range(0, y_val_discr.shape[2])]

            y_train_orig = [y_train_orig[:, :, i] for i in range(0, y_train_orig.shape[2])]
            y_test_orig = [y_test_orig[:, :, i] for i in range(0, y_test_orig.shape[2])]
            y_val_orig = [y_val_orig[:, :, i] for i in range(0, y_val_orig.shape[2])]

            y_train = [numpy.array(
                numpy.apply_along_axis(categorize, axis=1, arr=y, _output_bins=_output_bins), dtype='int8') for (y, _output_bins) in zip(y_train_discr, self._output_bins)]
            y_test = [numpy.array(
                numpy.apply_along_axis(categorize, axis=1, arr=y, _output_bins=_output_bins), dtype='int8') for (y, _output_bins) in zip(y_test_discr, self._output_bins)]
            y_val = [numpy.array(
                numpy.apply_along_axis(categorize, axis=1, arr=y, _output_bins=_output_bins), dtype='int8') for (y, _output_bins) in zip(y_val_discr, self._output_bins)]

        if self._include_rc:
            print('Adding reverse complement of genomic sequences and binding profiles for training, test and validation data... ')
            x_train, y_train_orig, y_train_smooth, y_train_discr, y_train = self._compute_reverse_complements(x_train, y_train_orig, y_train_smooth, y_train_discr, y_train)
            x_test, y_test_orig, y_test_smooth, y_test_discr, y_test = self._compute_reverse_complements(x_test, y_test_orig, y_test_smooth, y_test_discr, y_test)
            x_val, y_val_orig, y_val_smooth, y_val_discr, y_val = self._compute_reverse_complements(x_val, y_val_orig, y_val_smooth, y_val_discr, y_val)

        # to make pretty plots keeping the original scores
        self.train_y_data_smooth = y_train_smooth
        self.test_y_data_smooth = y_test_smooth
        self.val_y_data_smooth = y_val_smooth

        self.train_y_data_discr = y_train_discr
        self.test_y_data_discr = y_test_discr
        self.val_y_data_discr = y_val_discr

        self.train_y_data_orig = y_train_orig
        self.test_y_data_orig = y_test_orig
        self.val_y_data_orig = y_val_orig

        self.train_x_data = x_train
        self.test_x_data = x_test
        self.val_x_data = x_val

        self.train_y_data = y_train
        self.test_y_data = y_test
        self.val_y_data = y_val

        self._is_training_loaded = True

    def _load_data(self):
        """
        Creates the output directory and calls functions to load genome data and binding profile data from
        :param binding_profile_data_source: list of str, path to the binding profile data sets
        :return:
        """
        output_directory = os.path.join(self._runtime_output, "data_load")
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        self._load_genome_data(self._runtime_data_source)
        if len(self._binding_profile_data) > 0:
            return self._load_binding_profile_data(self._binding_profile_data)
        else:
            return list()

    def _load_data_timed(self):
        """
        Creates the output directory and calls functions to load genome data and binding profile data from
        :param binding_profile_data_source: list of str, path to the binding profile data sets
        :return:
        """
        output_directory = os.path.join(self._runtime_output, "data_load")
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        self._load_genome_data_timed(self._runtime_data_source)
        if len(self._binding_profile_data) > 0:
            return self._load_binding_profile_data_timed(self._binding_profile_data)
        else:
            return list()

    def _load_binding_profile_data_timed(self, runtime_data_source):
        """
        Timed self._load_genome_data()
        :return:
        """
        time_start = time.clock()
        self._load_binding_profile_data(runtime_data_source)
        print("Binding profile data was loaded in: " + str(time.clock() - time_start) + " seconds")

    def _load_genome_data_timed(self, runtime_data_source):
        """
        Timed self._load_genome_data()
        :return:
        """
        time_start = time.clock()
        self._load_genome_data(runtime_data_source)
        print("Genome data was loaded in: " + str(time.clock() - time_start) + " seconds")

    def _load_genome_data(self, runtime_data_source):
        """
        Loads genomic sequences from fasta files located in self._runtime_data_source. Allowed file formats are
        .fa, .fasta, and .fsa. Mitochondrial and X and Y chromosomes are not loaded.

        Parameters used
        ----------------
        self._runtime_processed_data_source - str with path to fasta file or to directory containing fasta files
        self.load_existing_data
        self.save_data

        Parameters returned
        ----------------
        self.genome_data: list of numpy arrays holding the one-hot encoded sequence data
        self.chr_info : list of sequence names in the same order as self.genome_data

        :return:
        """

        genome_dir = os.path.join(self._runtime_processed_data_source, 'genome_data')
        if os.path.exists(genome_dir) and self.load_existing_data:
            print("Loading pre-existing genomic data")
            sequences = glob.glob(os.path.join(genome_dir, "*.npy"))
            x_data = []
            # this is because if promoter sequences are loaded from the same fasta file the genome data is a matrix, need to remap it to list
            for s in sequences:
                ls = numpy.load(s)
                if ls.shape[0] > 1:
                    x_data.extend(numpy.squeeze(numpy.split(ls, ls.shape[0]), 0))
                else:
                    x_data.append(ls)
            sequences = glob.glob(os.path.join(genome_dir, "*.json"))
            chr_info = []
            for s in sequences:
                chr_info.extend(utils.load_json(s))

        else:
            print("Loading genomic data... ")
            if not os.path.exists(genome_dir):
                if self.save_data:
                    os.makedirs(genome_dir)
            x_data = []
            chr_info = []
            # Find the sequence files
            if os.path.isfile(runtime_data_source):
                sequences = [runtime_data_source]

            else:
                sequences = glob.glob(os.path.join(runtime_data_source, "*.fa"))
                sequences.extend(glob.glob(os.path.join(runtime_data_source, "*.fasta")))
                sequences.extend(glob.glob(os.path.join(runtime_data_source, "*.fsa")))
            # Loop through the sequences
            for sequence in sequences:
                fa_sequence = SeqIO.parse(sequence, 'fasta')
                fa_sequence = [str(fasta.seq) for fasta in fa_sequence]
                name = [fasta.name for fasta in SeqIO.parse(sequence, 'fasta')]
                # Format the sequence into a binarised form
                if len(fa_sequence) > 1 and len(name) == 1:
                    # if there is more than one fasta sequence in the file but only one name, we want to stack it into a numpy array.
                    #  to do this we need to pad
                    # everything so it can be stacked into a numpy array
                    max_len = max([len(seq) for seq in fa_sequence])
                    if self._pad_to_center:
                        def is_odd(num):
                            return num & 0x1

                        def get_pad_length(x_len, ml):
                            if is_odd(ml - x_len):
                                return (ml - x_len) / 2, (ml - x_len) / 2 + 1
                            else:
                                return (ml - x_len) / 2, (ml - x_len) / 2

                        sequence_matrix = numpy.array(
                            [utils.bases_to_binary(x, paddingL=get_pad_length(len(x), max_len)[0], paddingR=get_pad_length(len(x), max_len)[1]) for x in fa_sequence])
                    else:
                        # swap axis to have channels come last in x_data
                        sequence_matrix = numpy.array(
                            [utils.bases_to_binary(x, paddingL=0, paddingR=max_len - len(x)) for x in fa_sequence])
                else:
                    sequence_matrix = [utils.bases_to_binary(x) for x in fa_sequence]
                x_data.extend(sequence_matrix)
                if len(name) <= 1:
                    # if promoter sequences are loaded from the same fasta file the genome data is list of promoter sequences
                    # if there are not several fasta sequences  in the fasta file, then use the filename as name.
                    name, _ = os.path.splitext(os.path.basename(sequence))
                    name = [name]
                chr_info.extend(name)

                if self.save_data:
                    print("Saving genome data.")
                    output_prefix = os.path.join(genome_dir, os.path.splitext(os.path.basename(sequence))[0])
                    numpy.save(output_prefix + ".npy", sequence_matrix)
                    utils.save_json(chr_info, output_prefix + ".json")

        self.genome_data = x_data
        self.chr_info = chr_info

        if 'chrM' in self.chr_info:
            print('Mitochondrial chromosome will be kept as separate data...')
            self.chrM_genome_data = self.genome_data[self.chr_info.index('chrM')]
            self.genome_data = [self.genome_data[i] for i in range(len(self.genome_data)) if self.chr_info[i] != 'chrM']
            self.chr_info = [a for a in self.chr_info if a != 'chrM']

        if 'chrY' in self.chr_info:
            print('Chromosome Y will be kept as separate data...')
            self.chrX_genome_data = self.genome_data[self.chr_info.index('chrY')]
            self.genome_data = [self.genome_data[i] for i in range(len(self.genome_data)) if self.chr_info[i] != 'chrY']
            self.chr_info = [a for a in self.chr_info if a != 'chrY']
            if 'chrX' in self.chr_info:
                print('Assumimg that chr numbers are NOT roman numbers and ChrX is also kept as separate data...')
                self.chrX_genome_data = self.genome_data[self.chr_info.index('chrX')]
                self.genome_data = [self.genome_data[i] for i in range(len(self.genome_data)) if
                                    self.chr_info[i] != 'chrX']
                self.chr_info = [a for a in self.chr_info if a != 'chrX']

        self._is_genome_data_loaded = True

    def _validate_data_dir(self, profile_dir):

        if not os.path.exists(profile_dir):
            return False
        else:
            sequences = glob.glob(os.path.join(profile_dir, "*.pkl"))
            expected_files = [os.path.join(profile_dir, 'genome_data.pkl'), os.path.join(profile_dir, 'profiles_discr.pkl'),
                              os.path.join(profile_dir, "profiles_smooth.pkl"), os.path.join(profile_dir, "profiles.pkl")]
            if not numpy.all(numpy.array([x in sequences for x in expected_files])):
                print('Error: not all files exist. Re-loading.. ')
                return False
            params = utils.load_json(os.path.join(profile_dir, "preprocessing_params.json"))
            if not numpy.all(params == self.preprocessing_params):
                print('Pre-processing parameters have changed.  Re-processing with new parameters.. ')
                return False

        return True

    def _load_binding_profile_data(self, binding_profile_data_source):
        """
        Loads the binding profile data from


        Parameters used
        ----------------
        self.genome_data
        self.chr_info
        self.load_existing_data
        self.save_data
        self.preprocessing_params
        self.step_size

        Parameters returned
        ----------------
        self._genome_data - this is as self.genome_data but adjusted to any clipped regions in the binding profiles
        self.y_data - binding profile data
        self.x_data  - genome data in windows of same size as self.y_data

        self.y_data - smoothed data split into windows
        self.profile_data_split - original data but split into windows
        self.profile_data_smooth - smoothed original data
        self.y_data_discr - smoothed, discretised and split into windows
        self.x_data - genome data split into windows
        self.profile_data_clipped - original data but removed unmapped regions at start and end of chromosomes.



        :param binding_profile_data_source: list of str, directories of the binding profile data sets
        :return:
        """

        profile_dir = os.path.join(self._runtime_processed_data_source, 'binding_profile_data')

        load_ok = False
        if self.load_existing_data:
            load_ok = self._validate_data_dir(profile_dir)
            if load_ok:
                print("Loading existing profile data")
                _genome_data, profile_data_smooth_discr, profile_data_smooth, profile_data_clipped = self._load_existing_data(
                    profile_dir)

        if (not self.load_existing_data) or (not load_ok):
            # print('Loading and processing binding profile data...')
            # setting some defaults if no parameters are passed
            assert self.preprocessing_params is not None, 'No preprocessing parameters given, defaulting to '
            starts, ends, profile_data_smooth, profile_data_smooth_discr, \
            profile_data_clipped = self._load_profile_data(binding_profile_data_source)

            # removing genomic ranges at beginning and end of chromosome where we don't have a signal - this data is
            # needed to compare later binding profiles with original data
            self._genome_data = [self.genome_data[i][:, int(starts[self.chr_info[i]]):(int(ends[self.chr_info[i]]) + 1)]
                                 for i in range(len(self.chr_info))]

            _genome_data = self._genome_data

            if self.save_data:
                if not os.path.exists(profile_dir):
                    print('Saving data ... ')
                    os.makedirs(profile_dir)
                    utils.save_json(self.preprocessing_params, os.path.join(profile_dir, "preprocessing_params.json"))
                    utils.pickle_save(_genome_data, os.path.join(profile_dir, "genome_data.pkl"))
                    utils.pickle_save(profile_data_smooth_discr, os.path.join(profile_dir, "profiles_discr.pkl"))
                    utils.pickle_save(profile_data_smooth, os.path.join(profile_dir, "profiles_smooth.pkl"))
                    utils.pickle_save(profile_data_clipped, os.path.join(profile_dir, "profiles.pkl"))

        if self.fragment_length is not None:
            y_data = [self._split_binding_profile_data(tmp_discr) for tmp_discr in profile_data_smooth_discr]
            profile_data_split = [self._split_binding_profile_data(tmp) for tmp in profile_data_clipped]
            profile_data_split_smooth = [self._split_binding_profile_data(tmp_smooth) for tmp_smooth in profile_data_smooth]

            self.y_data = profile_data_split_smooth  # smoothed data split into windows
            self.profile_data_split = profile_data_split  # original data but split into windows
            self.y_data_discr = y_data  # smoothed, discretised and split into windows

            if self.step_size is not None:
                x_data = [numpy.squeeze(view_as_windows(x[:, :(x.shape[-1] // self.fragment_length * self.fragment_length)],
                                                        window_shape=(x.shape[0], self.fragment_length), step=self.step_size), axis=0) for x in _genome_data]
            else:
                x_data = [numpy.squeeze(view_as_blocks(x[:, :(x.shape[-1] // self.fragment_length * self.fragment_length)],
                                                       block_shape=(x.shape[0], self.fragment_length)), axis=0) for x in _genome_data]
        else:
            self.y_data = profile_data_smooth
            self.profile_data_split = profile_data_clipped
            y_data = profile_data_smooth_discr
            y_data = [[numpy.expand_dims(y, 0) for y in y_dat] for y_dat in y_data]
            x_data = [numpy.expand_dims(x, 0) for x in _genome_data]
            self.y_data_discr = y_data

        x_data = [numpy.swapaxes(x, 1, 2) for x in x_data]
        self.x_data = x_data  # genome data split into windows
        # assert everything is cut into the same number of fragments
        assert numpy.all([
            numpy.array_equal(numpy.array([x.shape[0] for x in x_data]),
                              numpy.array([y.shape[0] for y in y_dat])) for y_dat in y_data])

        self.profile_data_smooth = profile_data_smooth  # smoothed original data
        self.profile_data_clipped = profile_data_clipped  # original data but removed unmapped regions at start and end of chromosomes.
        self.n_output_features = len(self.y_data_discr)

        def classes_and_counts(arr):
            return numpy.unique(arr.flatten(), return_counts=True)

        # counting how often each class occurs in the dataset
        _class_counts = [classes_and_counts(numpy.concatenate([yd.flatten() for yd in y])) for y in self.y_data_discr]
        self._output_bins = [numpy.int(c[0].max()) + 1 for c in _class_counts]
        print('Binding profiles were binned into ' + str(self._output_bins) + ' bins.')
        self._are_profiles_loaded = True

    def _load_profile_data(self, binding_profile_data_source):
        """
        Loads each profile data, preprossses it by filling in NAs, normalising, smoothing, discretisation and finally
        splitting into windows.


        Parameters
        ----------
        self.preprocessing_params
        self.chr_info

        Calls
        -----
        self.pre_process()
        self._process_profile()
        self._split_binding_profile_data()


        :param binding_profile_data_source: list of strings, directories of the binding profile data sources
        :return: a lit of arrays:
        first position of each sequence covered by any reads, last position of each sequence covered by any reads,
        smoothed profiles split into windows, normalised profile without NAs split into windows, smoothed profile data,
        smoothed& discretised data split into windows, normalised profile without NA clipped
        """

        starts = []
        ends = []
        profile_data = []
        profile_data_clipped = []
        profile_data_smooth = []
        profile_data_smooth_discr = []
        # reading in profile and preprocessing
        for (profile, preprocessing_params) in zip(binding_profile_data_source, self.preprocessing_params):
            print('using pre-processing params:')
            print(preprocessing_params)
            # Load the profile
            seq_profile, seqname_column_name, start_column_name, end_column_name,score_column_name, start, end = self._load_profile(profile)
            # Preprocess -
            profile_data.append(self.pre_process(seq_profile, seqname_column_name, start_column_name, end_column_name,
                                                 score_column_name, preprocessing_params))
            starts.append(start)
            ends.append(end)
        # # trimming ends of chromosome where we don't have all profiles
        starts = pandas.concat(starts, axis=1).max(axis=1)
        ends = pandas.concat(ends, axis=1).min(axis=1)

        for (prof, preprocessing_params) in zip(profile_data, self.preprocessing_params):
            tmp = []
            for i in range(len(self.chr_info)):
                # trimm ends of binding profile wehre we don't have all profiles
                # ends is actual index of last entry, for range need to add one to include it
                tmp.append(prof[i][int(starts[self.chr_info[i]]):int(ends[self.chr_info[i]] + 1)])

            profile_data_clipped.append(tmp)

            # process binding profile (ie smooth and discretise)
            tmp_smooth, tmp_discr = self._process_profile(tmp, preprocessing_params, self.verbose)
            profile_data_smooth.append(tmp_smooth)
            profile_data_smooth_discr.append(tmp_discr)

        return starts, ends, profile_data_smooth, profile_data_smooth_discr, profile_data_clipped

    def _load_profile(self, binding_profile_data_source):
        """
        Loads a single profiles from .csv, .bed,  or .mat files. If input is a .csv file, the columns have to be in the
        following order: seqname, start, score (not necessarily with these column names). The unique seqnames in this
        and the .bed format are used as chromosome names. If a file in .mat format is passed, this should be the
        output of deeptools' ComputeMatrix generated with the command:
        'computeMatrix reference-point --referencePoint TSS -S bigwig_file -R regions_bed_file \
        -a downstream_bp -b upstream_bp --binSize 1 --sortRegions keep --nanAfterEnd  --missingDataAsZero --numberOfProcessors p \
        -o outfile.compMat.gz --outFileNameMatrix outfile.compMat.mat'

        Parameters used
        ----------------

        Parameters returned or changed
        ------------------------------
        self._using_full_genomic_sequences - if .mat set to False
        self.zero_based - if the positions are zero-based, if file format .mat or .bed set to True

        :param binding_profile_data_source: string - path to binding profile data
        :return:
        seq_profile: pandas dataframe holding the seqname, start and scores columns as provided in the input
        seqname_column_name: str, name of column of seq_profile holding the seqnames if available
        start_column_name: str, name of column of seq_profile holding the start position if available
        score_column_name: str, name of column of seq_profile holding the scores if available
        starts: pandas dataframe with the first covered position of each of the sequences with the seq names (as held in self.chr_info) as index
        ends: pandas dataframe with the last covered position of each of the sequences with the seq names (as held in self.chr_info) as index

        """
        filename, extension = os.path.splitext(binding_profile_data_source)
        # this assumnes a comma or tab delimeted coverage file that heas a per position score
        # col 0 :seqnames
        # col1 : start/positions
        # col2: score
        # missing scores will be imputed using chromsome wide background and are assumed to be absent rather than missing
        if extension == '.csv' or extension == '.bed':
            if extension == '.csv':
                print('I am assuming that the columns in the provided csv file are: seqname, start, score. If this is not the case, abort and change the input format!')
                seq_profile = pandas.read_csv(binding_profile_data_source)
                # assuming that seq info is in first col
                seqname_column_name = seq_profile.columns[0]
                # assuming that start pos is in 2nd col
                start_column_name = seq_profile.columns[1]
                # assuming that 3rd column is the score
                score_column_name = seq_profile.columns[2]
                end_column_name = None
            elif extension == '.bed':
                seq_profile = pandas.read_csv(binding_profile_data_source, sep='\t', header=None)
                # assuming that seq info is in first col
                seqname_column_name = seq_profile.columns[0]
                # assuming that start pos is in 2nd col
                start_column_name = seq_profile.columns[1]
                # assuming that end pos is in 3rd column
                end_column_name = seq_profile.columns[2]
                # assuming that 5th column is the score
                score_column_name = seq_profile.columns[4]
                # bed files are zero based
                self.zero_based = True
                # making sure names are strings
                seq_profile[seqname_column_name] = seq_profile[seqname_column_name].astype(str)
                # check if ch names are in roman numbers

            chr_names = set(seq_profile[seqname_column_name])
            # checking that there is not a mix of roman numbers
            if 'chrI' in chr_names and 'chr1' in self.chr_info:
                seq_profile[seqname_column_name] = seq_profile[seqname_column_name].apply(
                    lambda x: 'chr' + str(roman.fromRoman(x[3:])) if x[3:] != 'M' else x)

            starts = seq_profile.groupby(seqname_column_name)[start_column_name].min()
            ends = seq_profile.groupby(seqname_column_name)[start_column_name].max()
        else:
            if extension == '.mat':
                print('Assuming output matrix of ComputeMatrix of deeptools generated with the command:')
                print('computeMatrix reference-point --referencePoint TSS -S bigwig_file -R regions_bed_file \
                -a downstream_bp -b upstream_bp --binSize 1 --sortRegions keep --nanAfterEnd  --missingDataAsZero --numberOfProcessors p \
                -o outfile.compMat.gz --outFileNameMatrix outfile.compMat.mat')
                seq_profile = pandas.read_csv(binding_profile_data_source, sep='\t', skiprows=3, header=None)
                seq_profile.fillna(0, inplace=True)
                self._using_full_genomic_sequences == False
                self.zero_based = True
                seqname_column_name = None
                start_column_name = None
                end_column_name = None
                score_column_name = None
                starts = pandas.DataFrame([0] * seq_profile.shape[0])
                starts['chr_info'] = self.chr_info
                starts.set_index('chr_info', inplace=True)
                ends = pandas.DataFrame([seq_profile.shape[-1] - 1] * seq_profile.shape[0])
                ends['chr_info'] = self.chr_info
                ends.set_index('chr_info', inplace=True)
            else:
                print('Error: file format not implemented yet. please provide either csv or bed files.')

        return seq_profile, seqname_column_name, start_column_name, end_column_name, score_column_name, starts, ends

    def pre_process(self, profile_dataframe, seqname_column_name, start_column_name, end_column_name, score_column_name, preprocessing_params):
        """
        Preprocessed the raw data according to the passed preprocessing parameters.

        # assings unmapped bases the value preprocessing_params['assign_non-covered bases']
        # clips if raw data
        # normalises read counts if normalisation function is given

        Preprocessing parameters
        ------------------------
        times_median_coverage_max - values greater than this multiple of the median will be clipped
        normalise_read_counts - how should readcounts be normalised, choose from 'log2_ratio' to take the log2 ratio
        between the signal and the genome mean, or 'genome_mean' to subtract the genomic mean (negative values are set to zero)
        assign_non_covered_bases - values to impute missing values with, choose a value or from 'genome_mean',
        'genome_median', 'chrom_mean', if None defaults to chromosome mean.

        Parameters used
        ----------------
        self.data_format
        self._using_full_genomic_sequences
        self.zero_based
        self.genome_data
        self.chr_info

        Parameters returned or changed
        ------------------------------

        :param profile_dataframe: pandas dataframe holding the seqname, start and scores columns as provided in the input
        :param seqname_column_name: str, name of column of seq_profile holding the seqnames if available
        :param start_column_name: str, name of column of seq_profile holding the start position if available
        :param end_column_name: str, name of column of seq_profile holding the end position if available
        :param score_column_name: str, name of column of seq_profile holding the scores if available
        :param preprocessing_params: dictionary holding the names and values of the preprocessing parameters
        :return: List of the normalised and imputed binding profile data for each sequence
        """

        if self.verbose == 1:
            if self.data_format == 'raw_counts':
                print('Assuming raw data counts. Replacing missing values with genome average and constraining max values to ' + str(preprocessing_params['times_median_coverage_max']) + 'x median genomic coverage')
            elif self.data_format == 'processed_counts':
                print('Assuming processed coverage counts.')
                if 'normalise_read_counts' in preprocessing_params:
                    if preprocessing_params['normalise_read_counts'] is not None:
                        print("Are you sure you want to normalise with " + preprocessing_params['normalise_read_counts'] + '?')

        if self._using_full_genomic_sequences:
            if self.zero_based:
                starts = 0
                ends = {a: self.genome_data[self.chr_info.index(a)].shape[-1] - 1 for a in self.chr_info}
            else:
                starts = 1
                ends = {a: self.genome_data[self.chr_info.index(a)].shape[-1] for a in self.chr_info}

            # we need a value for each position, so we are just filling in NAs - using value=None this is automatically
            # chosen to be the chromosome mean.

            y_orig = [add_missing_elements(profile_dataframe[profile_dataframe[seqname_column_name] == a],
                                           seqnames_column_name=seqname_column_name,
                                           start_column_name=start_column_name,
                                           end_column_name=end_column_name,
                                           score_column_name=score_column_name, value=None,
                                           start=starts,
                                           end=ends[a])[score_column_name] for a in self.chr_info]

            def process(y, params, genome_mean, genome_median):

                if self.data_format == 'raw_counts':
                    y = numpy.clip(y, _EPSILON, preprocessing_params['times_median_coverage_max'] * genome_median)
                if 'normalise_read_counts' in params:
                    if params['normalise_read_counts'] is not None:
                        if params['normalise_read_counts'] == 'log2_ratio':
                            y = numpy.log2(y + _EPSILON / (genome_mean + _EPSILON))
                        elif params['normalise_read_counts'] == 'genome_mean':
                            y = y - genome_mean
                            y[y < 0] = 0
                        else:
                            print('Normalisation function for read counts not yet implemented')
                return y

            genome_median = numpy.median(numpy.concatenate(y_orig))
            genome_mean = numpy.mean(numpy.concatenate(y_orig))
            if preprocessing_params['normalise_read_counts'] == 'log2_ratio':
                print('Normalising read counts by taking the log2 ratio between read counts and the genome mean')
            elif preprocessing_params['normalise_read_counts'] == 'genome_mean':
                print('Normalising read counts by subtracting the genome mean')

            y_orig = [process(y, preprocessing_params, genome_mean, genome_median) for y in y_orig]


        else:
            # if reading eg promoter profile data, start and end info in the bed files probably doesnt run from 0 to len(sequence), so ignoring the start info and relying that
            # data has been correctly processed and matches the lengths in the genome data.
            print('Please ensure that data is preprocessed so that profiles are right-padded (with mean or zeros as appropriate) to the max length of the provided genomic data')

            if score_column_name is not None:
                genome_mean = numpy.float(numpy.mean(profile_dataframe[[score_column_name]]))
                genome_median = numpy.float(numpy.median(profile_dataframe[[score_column_name]]))

                def process2(ser):
                    params = preprocessing_params

                    y = ser[score_column_name]
                    # y[y==0] = genome_mean
                    if self.data_format == 'raw_counts':
                        y = numpy.clip(y, _EPSILON, params['times_median_coverage_max'] * genome_median)
                    if 'normalise_read_counts' in params:
                        if params['normalise_read_counts'] is not None:
                            if params['normalise_read_counts'] == 'log2_ratio':
                                y = numpy.log2(y + _EPSILON / (genome_mean + _EPSILON))
                            elif params['normalise_read_counts'] == 'genome_mean':
                                y = y - genome_mean
                                y[y < 0] = 0
                            else:
                                print('Normalisation function for read counts not yet implemented')
                    return y

                if preprocessing_params['normalise_read_counts'] == 'log2_ratio':
                    print('Normalising read counts by taking the log2 ratio between read counts and the genome mean')
                elif preprocessing_params['normalise_read_counts'] == 'genome_mean':
                    print('Normalising read counts by subtracting the genome mean')
                if self.parallelize:
                    def applyParallel(dfGrouped, func):
                        retLst = Parallel(n_jobs=multiprocessing.cpu_count())(
                            delayed(func)(group) for name, group in dfGrouped)
                        return pandas.concat(retLst)

                    print('using parallel version: ')
                    y_orig = applyParallel(
                        profile_dataframe[[seqname_column_name, score_column_name]].groupby(seqname_column_name), process2)
                else:
                    y_orig = profile_dataframe[[seqname_column_name, score_column_name]].groupby(seqname_column_name).apply(
                        process2)
                y_orig = [numpy.array(y_orig[[a]]) for a in self.chr_info]


            else:
                # if score_column_name is none then we've been loading output of appropriately sorted computationMatrix output
                def process(y, params, genome_mean, genome_median):
                    y[y == 0] = genome_mean
                    if self.data_format == 'raw_counts':
                        y = numpy.clip(y, _EPSILON, preprocessing_params['times_median_coverage_max'] * genome_median)
                    if 'normalise_read_counts' in params:
                        if params['normalise_read_counts'] is not None:
                            if params['normalise_read_counts'] == 'log2_ratio':
                                y = numpy.log2(y + _EPSILON / (genome_mean + _EPSILON))
                            elif params['normalise_read_counts'] == 'genome_mean':
                                y = y - genome_mean
                                y[y < 0] = 0
                            else:
                                print('Normalisation function for read counts not yet implemented')
                    return y

                df = numpy.array(profile_dataframe)
                genome_median = numpy.median(df)
                genome_mean = numpy.mean(df)
                if preprocessing_params['normalise_read_counts'] == 'log2_ratio':
                    print('Normalising read counts by taking the log2 ratio between read counts and the genome mean')
                elif preprocessing_params['normalise_read_counts'] == 'genome_mean':
                    print('Normalising read counts by subtracting the genome mean')

                y_orig = numpy.apply_along_axis(process, 0, df, preprocessing_params, genome_mean, genome_median)
                y_orig = numpy.split(numpy.array(y_orig), y_orig.shape[0])
                y_orig = [numpy.squeeze(y) for y in y_orig]

        return y_orig

    def _process_profile(self, tmp, preprocessing_params, verbose=1):
        """
        Smoothes and discretises the binding profile data.

        Calls
        -----
        self._split_binding_profile_data
        self._smooth_profile_data
        self._discretize_binding_profile_data

        :param tmp: list of numpy arrays to be smoothed and discretised
        :param preprocessing_params: dictionary of preprocessing parameters containing 'u'
        :param verbose: 0 or 1
        :return: list of numpy arrays: smoothed input, discretised input
        """
        # splitting original data into windows
        # y_d_orig = self._split_binding_profile_data(tmp)
        # smooth data if requested
        if preprocessing_params['smooth_signal']:
            tmp = self._smooth_profile_data(tmp, preprocessing_params, self.verbose)
        # splitting smoothed data into windows
        # y_d_smooth = self._split_binding_profile_data(tmp)
        # discretise data
        if verbose == 1:
            print('Discretising smoothed signal with float_to_int with parameter u = ', preprocessing_params['u'])
        tmp_discr = self._discretize_binding_profile_data(tmp, preprocessing_params['u'])
        # splitting discretised data into windows
        # y_d = self._split_binding_profile_data(tmp_discr)
        return tmp, tmp_discr

    def _discretize_binding_profile_data(self, y, u):
        """
        Discretises a continuous signal by shifting it first by its minimum if the signal has negative values,
        then streching the signal by multiplication with u and then discretizing the signal using the floor
        function.

        :param y: list of numpy arrays to be discretised
        :param u: strech parameter
        :return: list of discretised numpy arrays
        """

        return [numpy.array(float_to_int(shift_positive(copy(seq)), u=u)) for seq in y]

    def _split_binding_profile_data(self, y_data):
        """
        Splits a numpy array into windows of the size self.fragment_length with stepsize self.step_size (if None, then
        no overlapping windows).

        Parameters
        ----------
        self.step_size
        self.fragment_length

        :param y_data: list of numpy arrays
        :return: list of numpy arrays that are now windows with step size from input data.
        """

        if self.step_size is not None:
            y_data_split = [numpy.squeeze(
                numpy.squeeze(view_as_windows(
                    numpy.expand_dims(y[:(y.shape[-1] // self.fragment_length * self.fragment_length)], axis=0),
                    window_shape=(1, self.fragment_length), step=self.step_size), axis=0), axis=1) for y in y_data]

        else:
            y_data_split = [numpy.squeeze(
                numpy.squeeze(view_as_blocks(
                    numpy.expand_dims(y[:(y.shape[-1] // self.fragment_length * self.fragment_length)], axis=0),
                    block_shape=(1, self.fragment_length)),
                    axis=0), axis=1) for y in y_data]
        return y_data_split

    def invert_discretizing(self, y):
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

    def _smooth_discrete_data(self, y, preprocessing_params):
        """
        smoothes a discretised numpy array

        :param y: discrete numpy array
        :param u: the parameter used to strecht the signal originally
        :return:
        """
        u = preprocessing_params['u']
        y = [numpy.array(int_to_float(seq, u)) for seq in y]
        y = numpy.vstack([numpy.array(gaussian_filter1d(seq, sigma=3, truncate=4.0)) for seq in y])
        return y

    def _smooth_profile_data(self, orig_target_scores, preprocessing_params, verbose=1):
        """
        Smoothes a continuous signal with a Savitzky-Golay filter or a Guassian 1D filter.
        Parameters
        ----------
        self.plot_fit_diganostics - if grisearch is performed should the diagnostics being plotted? recommended for
        first time but will take a long time.

        :param orig_target_scores:
        :param preprocessing_params: dictionary or parameters, needed are
        'smoothing_function' -  choose from 'savgol_filter' or 'gaussian_filter1d'
        'window_length' - sindow length 'savgol_filter is chosen, if None a grid search will be performed
        'polyorder' - order of polynomial if savgol filter is chosen, if None a grid search will be performed
        'sigma' - sd for 'gaussian_filter1d', if None a grid search will be performed
        'truncate' - truncation of 'gaussian_filter1d, defaults to 4.0

        :param verbose:
        :return:
        """

        if preprocessing_params['smoothing_function'] == 'savgol_filter':
            if verbose == 1:
                print('Using Savitzky-Golay Filter to smooth the data')
            if preprocessing_params['window_length'] is None and preprocessing_params['polyorder'] is None:
                print(
                    'Searching for best window length and polyorder via GridSearch with cross-validation. This will take a few minutes...')
                param, _ = gridsearchCV_preprocessing_params(orig_target_scores)
                print('Found best parameter combiation:')
                print(param)
                preprocessing_params.update(param)
            # smoothing original data
            orig_target_scores_smooth = [numpy.array(savgol_filter(copy(seq), window_length=preprocessing_params['window_length'], polyorder=preprocessing_params['polyorder'])) for seq in orig_target_scores]
        elif preprocessing_params['smoothing_function'] == 'gaussian_filter1d':
            if verbose == 1:
                print('Using Gaussian Kernel Filter to smooth the data')
            if preprocessing_params['sigma'] is None:
                print(
                    'Searching for best standard deviation and truncate value via GridSearch with cross-validation. This will take a few minutes...')
                param, _ = gridsearchCV_preprocessing_params(orig_target_scores, param_grid={"sigma": range(5, 30), "truncate": range(3, 6)}, smoother='gaussian_1d')
                print('Found best parameter combiation:')
                print(param)
                preprocessing_params.update(param)
                self.plot_fit_diganostics = True
            if not "truncate" in preprocessing_params:
                # setting to default of gaussian_filter1d
                preprocessing_params['truncate'] = 4.0
            orig_target_scores_smooth = [numpy.array(gaussian_filter1d(copy(seq), sigma=preprocessing_params['sigma'], truncate=preprocessing_params['truncate'])) for seq in orig_target_scores]
        else:
            print('Smoothing function not implemented, no smoothing used.')
            orig_target_scores_smooth = orig_target_scores

        if self.plot_fit_diganostics:
            print('Saving residual test plots to file.')
            plot.plot_fit_diagnostics(orig_target_scores, orig_target_scores_smooth, output_path=os.path.join(self._runtime_output, "data_load"),
                                      smoothing_function=preprocessing_params['smoothing_function'], plot_name='Residual_Tests')

        return orig_target_scores_smooth

    def _load_existing_data(self, profile_dir):
        """
        Loads exisitng data

        :param profile_dir: directory containing the saved data
        :return: the loaded data is in the following order:
        the one-hot encoded sequence data split into windows, the discretised binding data split into windows, the
        smoothed binding data split into windows, the original data split into windows, the smoothed originial binding
        data, and the original binding data.
        """

        print("Loading existing profile data")
        _genome_data = utils.pickle_load(os.path.join(profile_dir, "genome_data.pkl"))
        profile_discr = utils.pickle_load(os.path.join(profile_dir, "profiles_discr.pkl"))
        profile_smooth = utils.pickle_load(os.path.join(profile_dir, "profiles_smooth.pkl"))
        profile = utils.pickle_load(os.path.join(profile_dir, "profiles.pkl"))

        return _genome_data, profile_discr, profile_smooth, profile

    def _compute_class_weights(self, _class_counts, fragment_length):
        """
        Computes the class weights for the list of class counts obtained using numpy.unique(a, counts=True).
        Class frequencies are computed as class count divided by fragment_length. This is adjusted by dividing
        additionally by 2 if the reverse complement is included. Finally class frequencies are clipped at
        self.class_weight_cap. Plots of the calss counts, the class weights and the counts*weights are saved
        in "data_load" in the self._runtime_output directory.

        Parameters
        ----------
        self.class_weight_cap
        self._include_rc
        self.n_output_features

        Parameters returned/changed
        ------------------
        self.class_counts
        self._output_bins
        self.class_frequencies

        :param _class_counts: list off class counts obtained using numpy.unique(a, counts=True)
        :param fragment_length: self.fragment_length oir, if None, the length of the input sequences
        :return:
        """

        self.class_counts = [c[1] for c in _class_counts]

        if not numpy.all(numpy.array([bin == c.shape[0] for bin, c in zip(self._output_bins, self.class_counts)])):
            print('Not all classes are represented in the data, this is not very efficient. Consider re-running using a smaller \'times_median_coverage_max\'- preprocessing parameter')

            def add_missing_class_counts(_class_counts_item):
                missing_indices = missing_elements(_class_counts_item[0].tolist(), start=0, end=numpy.int(_class_counts_item[0].max()))
                idx = [m - i * 1 for m, i in zip(missing_indices, range(len(missing_indices)))]
                return numpy.insert(_class_counts_item[1], idx, 0, axis=0)

            self.class_counts = [add_missing_class_counts(c) for c in _class_counts]

        self.class_frequencies = [total_class_counts / float(fragment_length) for total_class_counts in self.class_counts]
        if self._include_rc:
            self.class_frequencies = [class_frequencies / 2. for class_frequencies in self.class_frequencies]
        if not isinstance(self.class_weight_cap, list):
            self.class_weight_cap = [self.class_weight_cap]
        self.class_weight = [numpy.min(numpy.vstack(
            (numpy.median(class_frequencies) / (class_frequencies), class_weight_cap * numpy.ones_like(class_frequencies))), axis=0) for (class_frequencies, class_weight_cap) in zip(self.class_frequencies, self.class_weight_cap)]
        output_directory = os.path.join(self._runtime_output, "data_load")
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        pandas.concat([pandas.DataFrame(c) for c in self.class_counts], axis=1).plot(subplots=True, title='Class counts')
        pyplot.savefig(os.path.join(output_directory, "Class_counts.png"))
        pyplot.close()

        pandas.concat([pandas.DataFrame(c) for c in self.class_weight], axis=1).plot(subplots=True, title='Class weights')
        pyplot.savefig(os.path.join(output_directory, "Class_weights.png"))
        pyplot.close()

        pandas.concat([pandas.DataFrame(self.class_counts[i] * self.class_weight[i]) for i in range(self.n_output_features)], axis=1).plot(subplots=True, title='Class counts * class weights')
        pyplot.savefig(os.path.join(output_directory, "Class_counts_weights.png"))
        pyplot.close()

    def _pad_DNA(self, paddingL, paddingR):
        """
        Pads genome data held in self.x_data and self.test_x_data from left of right with N's represented with an
        0.25* all-one-vector

        :param paddingL: how many bp to pad from left
        :param paddingR: how many bp to pad from right
        :return:
        """
        # In training we just have a numpy array, in prediction we have a list of them (which may be length 1)
        if isinstance(self.x_data, (numpy.ndarray, numpy.generic)):
            self.x_data = utils.padding_dna(self.x_data, paddingL, paddingR)
        else:
            tmp = []
            for sequence_windows in self.x_data:
                tmp.append(utils.padding_dna(sequence_windows, paddingL, paddingR))
            self.x_data = tmp
        if self.val_x_data is not None:
            self.val_x_data = utils.padding_dna(self.val_x_data, paddingL, paddingR)

    def _compute_GC_content(self, data=None):
        """
        Comptes the GC content of data (if not None), otherwise of self.x_data.
        :param data: one-hot-encoded genomic data to compute GC content. Defaults to None.
        :return:
        """
        if data is not None:
            return numpy.sum(data, axis=(1, 0), dtype='float32') / (
                    data.shape[0] * data.shape[1])
        elif self.x_data is not None:
            return numpy.sum(self.x_data, axis=(1, 0), dtype='float32') / (
                    self.x_data.shape[0] * self.x_data.shape[1])
        else:
            print('Can not compute GC content without any data.')
            return

    def _compute_reverse_complements(self, x_data, y_orig, y_smooth, y_discr, y_data):
        """
        Computes the reverse complement for each sequence in x_data and reversed the binding data held in y_orig, y_smooth
        y_discr and y_data accordingly to match the orientation of x_data.
        :param x_data: list or array of one-hot encoded sequence data to be reverse complemented
        :param y_orig: list of original binding profile data
        :param y_smooth: list of smoothed originial binding profile data
        :param y_discr: list of discretised bidning profile data
        :param y_data: list of discretised and windowed binding profile data.
        :return:
        """

        x_data = numpy.vstack((x_data, numpy.array(
            [numpy.swapaxes(utils.reverse_complement(numpy.swapaxes(seq, 1, 0)), 1, 0) for seq in x_data])))

        y_orig = [numpy.vstack((y, numpy.flip(y, 1))) for y in y_orig]

        y_smooth = [numpy.vstack((y, numpy.flip(y, 1))) for y in y_smooth]

        y_discr = [numpy.vstack((y, numpy.flip(y, 1))) for y in y_discr]

        y_data = [numpy.vstack((y, numpy.flip(y, 1))) for y in y_data]

        return x_data, y_orig, y_smooth, y_discr, y_data

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
