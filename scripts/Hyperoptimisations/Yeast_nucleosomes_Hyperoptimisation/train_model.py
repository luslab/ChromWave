import matplotlib
matplotlib.use('Agg')
import os
import numpy
from hyperopt import STATUS_OK

from chromwave.functions import utils
from chromwave import chromwavenet, runtime_dataset, filesystem
import pickle
import tensorflow as tf
import argparse
import sys


from sklearn.model_selection import StratifiedKFold,KFold

def main(args,w_array):
    # # Build the dataset


    # Build the runtime dataset and catch exceptions here

    ###########################################
    # The model
    #########################################
    print('Assigning all model parameters')
    model = chromwavenet.ChromWaveNet()
    model.penalties = w_array
    model._pool_pretrained_conv = ['global'] * (
                len(args.keras_models) + len(args.numpy_models) + len(args.wavenet_models))
    model.early_stopping_patience = args.early_stopping_patience
    model.early_stopping = False
    model.penalise_misclassification = False
    model.batch_normalization = True
    model.use_skip_connections = True
    model.optimizer = args.optimizer
    if args.amsgrad == 'True':
        model.amsgrad = True
    else:
        model.amsgrad = False
    pool_sizes = [int(p) if not p in {'global', 'global_max', 'global_max', None} else p for p in args.pool_sizes]
    model.pool_sizes = pool_sizes
    model.kernel_lengths = args.kernel_lengths
    model.conv_n_filters = args.conv_n_filters
    model.dilation_depth = args.dilation_depth
    model.dilation_kernel_size = args.dilation_kernel_size
    model.learning_rate = args.learning_rate
    model.dropout = args.dropout
    model.res_dropout = args.res_dropout
    model.res_l2 = args.res_l2
    model.final_l2 = args.final_l2
    model.reduceLR_rate = args.reduceLR_rate
    model.minimum_batch_size = args.minimum_batch_size
    model.reduceLR_rate = args.reduceLR_rate
    model.max_epochs = args.max_epochs

    print('building model')
    pretrained_model_path = {'keras': args.keras_models, 'numpy': args.numpy_models, 'chromwave': args.wavenet_models}
    model.build(run_dataset, pretrained_model_path, weight_classes=True)
    print('fitting model')
    model_toFit = model._neural_network
    model_toFit.compile(optimizer=model.optim, loss=model.loss, metrics=model.all_metrics)
    model.serialize_json(output_filepath=os.path.join(f.get_output_directory(), 'WaveNet_Setup.json'))

    try:
        print('Start training')

        if args.use_crossvalidation:
            print('Doing 3-fold cross-validation on the training data...')
            kfold = KFold(n_splits=3, shuffle=True, random_state=32)
            cvscores = []
            for train, test in kfold.split(range(run_dataset.train_x_data.shape[0])):
                print('Cross-fold ' + str(len(cvscores)+1))+'/3'
                train_x_data = run_dataset.train_x_data[train]
                train_y_data = [y[train] for y in run_dataset.train_y_data]
                test_x_data = run_dataset.train_x_data[test]
                test_y_data = [y[test] for y in run_dataset.train_y_data]

                training_history = model_toFit.fit(x=train_x_data,
                                                   y=train_y_data,
                                                   epochs=model.max_epochs,
                                                   callbacks=model.callbacks,
                                                   verbose=0,
                                                   batch_size=model.minimum_batch_size
                                                   )

                # evaluate the model
                scores = model_toFit.evaluate(test_x_data, test_y_data, verbose=0)
                cvscores.append(scores)
        else:
            train_x_data = run_dataset.train_x_data
            train_y_data = run_dataset.train_y_data
            test_x_data = run_dataset.test_x_data
            test_y_data = run_dataset.test_y_data

            training_history = model_toFit.fit(x=train_x_data,
                                               y=train_y_data,
                                               epochs=model.max_epochs,
                                               callbacks=model.callbacks,
                                               validation_data=(
                                               test_x_data, test_y_data),
                                               verbose=0,
                                               batch_size=model.minimum_batch_size
                                               )
        if args.use_crossvalidation:
            loss_0, acc_0, mse_0, pcor_0 = numpy.mean(numpy.array(cvscores), axis=0)

            # print("%.2f%% (+/- %.2f%%)" % (numpy.mean(numpy.array(cvscores),axis=0), numpy.std(numpy.array(cvscores),axis=0)))
        else:
            model._neural_network = model_toFit
            model.training_history = training_history
            loss_0, acc_0, mse_0, pcor_0 = model_toFit.evaluate(test_x_data, test_y_data, verbose=0)

        results = {'loss': -0.5*(acc_0 + pcor_0), 'status': STATUS_OK}
        # print(results)
        utils.save_json(results, os.path.join(f.get_output_directory(), 'hyperopt_result.json'))

        print('Accuracy:', acc_0)

        print('Mean  squared error:', mse_0)

        print('Mean correlation coef:', pcor_0)

        print("Saving model to file ... ")
        model.save_model(f.get_output_directory(), 'BestWaveNet')

        with open(os.path.join(f.get_output_directory(), 'trainHistoryDict.pkl'), 'wb') as file_pi:
            pickle.dump(model.training_history.history, file_pi)

        model_json = model._neural_network.to_json()
        with open(os.path.join(f.get_output_directory(), "model.json"), "w") as json_file:
            json_file.write(model_json)
        # # serialize weights to HDF5
        model.save_weights_to(f.get_output_directory(), "BestWaveNet")

        if not os.path.exists(os.path.join(f.get_output_directory(), 'plots')):
            os.makedirs(os.path.join(f.get_output_directory(), 'plots'))

        print('Plotting training history...')
        model.plot_all_training_history(run_dataset, os.path.join(f.get_output_directory(), 'plots'))


    except Exception as ex:
        print("Failure to train a valid neural network for your data.")
        print(ex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate neural network for DNA-binding factor.')
    parser.add_argument(
        '--output_base',
        '-o',
        default='.',
        required=True,
        help='Output directory (defaults to current directory)'
    )
    parser.add_argument(
        '--genome_dir',
        '-tr',
        default=None,
        help='File path of source data. '
    )
    parser.add_argument(
        '--profiles',
        '-te',
        default=None,
        nargs='+',
        help='File paths of profile data'
    )
    parser.add_argument(
        '--optimizer',
        '-opt',
        default='adam',
        help='optimiser to use '
    )
    parser.add_argument(
        '--amsgrad',
        '-amsg',
        default='False',
        help='If using adam, should you use amsgrad? '
    )
    parser.add_argument(
        "--kernel_lengths",
        "-ckl",
        default=[24],
        type=int,
        nargs='+',
        help="The kernel lengths to use in the first convolutional layers (multiple integers separated by spaces)."
    )
    parser.add_argument(
        "--fragment_length",
        "-fs",
        default=2000,
        type=int,
        help="The fragment size for runtime data."
    )
    parser.add_argument(
        "--step_size",
        "-ss",
        default=None,
        type=int,
        help="The step size for runtime data."
    )
    parser.add_argument(
        '--keras_models',
        '-pkm',
        default=[],
        required=False,
        nargs='+',
        help='Direcotries pointing to pretrained keras models to be loaded into the net'
    )
    parser.add_argument(
        '--numpy_models',
        '-pdm',
        default=[],
        required=False,
        nargs='+',
        help='Direcotries pointing to numpy arrays of pretrained convolutional layers to be loaded into the net'
    )
    parser.add_argument(
        '--wavenet_models',
        '-wave',
        default=[],
        required=False,
        nargs='+',
        help='Direcotries pointing to pretrained wavenet models to be loaded into the net'
    )
    parser.add_argument(
        "--conv_n_filters",
        "-cnf",
        default=[60],
        type=int,
        nargs='+',
        help="The number of filters to use in the first convolutional layers (multiple integers separated by spaces)."
    )
    parser.add_argument(
        "--pool_sizes",
        "-cps",
        default=['global'],
        nargs='+',
        help="""The size of the region to pool in each pooling layer, positioned after each convolutional layer
        (multiple integers separated by spaces). 'Global' forces a global max pooling."""
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        default=0.001,
        type=float,
        help="""Learning rate""")

    parser.add_argument(
        "--early_stopping_patience",
        "-esp",
        default= 1,
        type=float,
        help="""early_stopping_patience""")

    parser.add_argument(
        "--dropout",
        "-do",
        default=0.3,
        type=float,
        help="""Dropout probability""")

    parser.add_argument(
        "--res_dropout",
        "-rdo",
        default=0.1,
        type=float,
        help="""Dropout probability of the residual block""")

    parser.add_argument(
        "--res_l2",
        "-rl2",
        default=0.001,
        type=float,
        help="""L2 regularisation for residual block""")

    parser.add_argument(
        "--final_l2",
        "-l2",
        default=0.001,
        type=float,
        help="""L2 regularisation""")

    parser.add_argument(
        "--reduceLR_rate",
        "-rlr",
        default=None,
        type=float,
        help="""Fraction the Learning rate is being reduced""")

    parser.add_argument(
        "--dilation_depth",
        "-dd",
        default=9,
        type=int,
        help="""Dilation depth""")

    parser.add_argument(
        "--dilation_kernel_size",
        "-dks",
        default=2,
        type=int,
        help="""Kernel size of the dilated convolutions""")

    parser.add_argument(
        "--weights_min",
        "-wmin",
        default=[0.8],
        nargs='+',
        type=float,
        help="""Penalise weight min """)

    parser.add_argument(
        "--weights_max",
        "-wmax",
        default=[1.2],
        nargs='+',
        type=float,
        help="""Penalise weight max """)


    parser.add_argument(
                        "--minimum_batch_size",
                        "-mbs",
                        default=32,
                        type=int,
                        help="The number of training examples in a mini_batch. Defaults to 32."
                        )
    parser.add_argument(
                        "--max_epochs",
                        "-me",
                        default=22,
                        type=int,
                        help="The number of epochs to be completed."
                        )

    parser.add_argument(
                        "--which_gpu",
                        "-wgpu",
                        default=0,
                        type=int,
                        help="Which GPU should be used? "
                        )

    parser.add_argument(
                        "--use_crossvalidation",
                        "-cv",
                        action='store_true',
                        default=False,
                        help="Should training be done in 3fold crossvalidation?"
                        )
    # Get args

    args = sys.argv[1:]
    args = parser.parse_args(args)

    print('Using GPU ' +str(args.which_gpu))
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    device = gpu_devices[args.which_gpu]
    tf.config.experimental.set_visible_devices(
        device,'GPU'
    )

    tf.config.experimental.set_memory_growth(device, True)
    print('Starting to train model .. ')

    f = filesystem.FileSystem(args.genome_dir, args.output_base, source_profile=args.profiles, overwrite=False,
                               test_fraction = 0.1, val_fraction=0.2, resume=False)
    run_dataset = runtime_dataset.RuntimeDataset(f)
    run_dataset._set_seed = 32
    run_dataset._shuffle_sequences=True
    run_dataset.save_data=False
    run_dataset._include_rc=True
    run_dataset.class_weight_cap = [40]
    run_dataset._remove_unmapped_training_regions = [0.3]
    run_dataset.data_format = 'processed_counts'
    nuc_preprocessing_params = {'times_median_coverage_max': None,
                                 'u': 2.5,
                                'smooth_signal': True,
                                'sigma': 5,
                                'truncate': 3,
                                'smoothing_function': 'gaussian_filter1d',
                                'normalise_read_counts': None}

    run_dataset.preprocessing_params = [nuc_preprocessing_params]
    runtime_dataset.fragment_length = args.fragment_length
    run_dataset.step_size = args.step_size

    try:
        run_dataset.load_data()
        weight_logspace_min = args.weights_min * run_dataset.n_output_features
        weight_logspace_max = args.weights_max * run_dataset.n_output_features
        w_arrays = [numpy.ones((_output_bins, _output_bins)) for _output_bins in run_dataset._output_bins]
        zs = [int(numpy.ceil(_output_bins / 2.)) for _output_bins in run_dataset._output_bins]

        for (w_array, z, w_min, w_max) in zip(w_arrays, zs, weight_logspace_min, weight_logspace_max):
            w_array[:z, -z:] = numpy.logspace(w_min, w_max, num=z)
            w_array[-z:, :z] = numpy.logspace(w_max, w_min, num=z)

        main(args, w_arrays)

    except Exception as ex:
        print("Failure to load data, this is a fatal exception")
        print(ex)



