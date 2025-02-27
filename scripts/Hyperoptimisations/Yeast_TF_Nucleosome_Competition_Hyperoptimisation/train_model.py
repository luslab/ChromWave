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


def main(args,w_array):
	# # Build the dataset


	# Build the runtime dataset and catch exceptions here

	###########################################
	# The model
	#########################################
	model = chromwavenet.ChromWaveNet()
	model.penalties = w_array
	model._pool_pretrained_conv = ['global']*(len(args.keras_models)+len(args.deepbind_models)+len(args.wavenet_models))
	model.inject_pwm_dir = args.pwm_dir
	model.inject_pwm_selection = args.inject_pwm_select
	model.early_stopping_patience = args.early_stopping_patience
	model.early_stopping = False
	model.penalise_misclassification = True
	model.batch_normalization = True
	model.use_skip_connections = True
	model.optimizer=args.optimizer
	if args.amsgrad=='True':
		model.amsgrad = True
	else:
		model.amsgrad = False
	# if there are too many conv filters we get cuda out of memory error if not global pooling.
	#pool_sizes = [p if n_conv<150 else 'global' for (p,n_conv)  in zip(args.pool_sizes, args.conv_n_filters) ]
	pool_sizes = [int(p) if not p in {'global', 'global_max','global_max',None} else p for p in args.pool_sizes ]
	model.pool_sizes = pool_sizes
	model.kernel_lengths = args.kernel_lengths
	model.conv_n_filters = args.conv_n_filters
	model.dilation_depth = args.dilation_depth
	model.dilation_kernel_size = args.dilation_kernel_size
	model.n_stacks = numpy.ceil(
	   (run_dataset.fragment_length - 1) / (2 ** model.dilation_depth * model.dilation_kernel_size - 1)).astype('int')
	model.learning_rate =  args.learning_rate
	model.dropout = args.dropout
	model.res_dropout = args.res_dropout
	model.res_l2 = args.res_l2
	model.final_l2 = args.final_l2
	model.reduceLR_rate=args.reduceLR_rate
	model.minimum_batch_size = args.minimum_batch_size
	model.reduceLR_rate =args.reduceLR_rate
	model.max_epochs=args.max_epochs
	model._use_multi_gpu = args.use_multi_gpu
	model._train_pretrained_conv = args.train_pretrained_layers
	print('building model')
	pretrained_model_path = {'keras': args.keras_models, 'deepbind': args.deepbind_models, 'wavenet': args.wavenet_models}
	model.build(run_dataset, pretrained_model_path,weight_classes=True)
	print('fitting model')

	model_toFit=model._neural_network
	model_toFit.compile(optimizer=model.optim, loss=model.loss, metrics=model.all_metrics)
	model.serialize_json(output_filepath=os.path.join( f.get_output_directory(),'WaveNet_Setup.json'))

	try:
		print('Start training')
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

		model._neural_network = model_toFit
		model.training_history = training_history
		loss_0, loss_1, loss_combined, acc_0, mse_0, pcor_0, acc_1, mse_1, pcor_1 = model_toFit.evaluate(test_x_data, test_y_data, verbose=0)
		with open(os.path.join(f.get_output_directory(), 'trainHistoryDict.pkl'), 'wb') as file_pi:
			pickle.dump(model.training_history.history, file_pi)
		print('Test accuracy 0:', acc_0)
		print('Test accuracy 1:', acc_1)
		print('Test mean  squared error 0:', mse_0)
		print('Test mean  squared error 1:', mse_1)
		print('Test mean correlation coef 0:', pcor_0)
		print('Test mean correlation coef 1:', pcor_1)
		print("Saving model to file ... ")
		model.save_model(f.get_output_directory(), 'BestWaveNet')

		model_json = model._neural_network.to_json()
		with open(os.path.join(f.get_output_directory(), "model.json"), "w") as json_file:
			json_file.write(model_json)
		# # serialize weights to HDF5
		model.save_weights_to(f.get_output_directory(), "BestWaveNet")

		if not os.path.exists(os.path.join(f.get_output_directory(), 'plots')):
			os.makedirs(os.path.join(f.get_output_directory(), 'plots'))


        print('Plotting training history...')
        model.plot_all_training_history(run_dataset, os.path.join(f.get_output_directory(), 'plots'))

        results = {'loss': -0.5*(acc_0+acc_1)-0.5*(pcor_0+pcor_1), 'status': STATUS_OK}

        utils.save_json(results, os.path.join(f.get_output_directory(), 'hyperopt_result.json'))
    #
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
        '--train_pretrained_layers',
        '-trprela',
        default='True',
        help='Should pretrained weights further trained or frozen? '
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
        "--step_size",
        "-ss",
        default=None,
        type=int,
        help="The step size for runtime data."
    )
    parser.add_argument(
        "--fragment_length",
        "-fl",
        default=2000,
        type=int,
        help="The fragment length for runtime data."
    )
    parser.add_argument(
        '--pwm_dir',
        '-pwm_dir',
        default=None,
        required=False,
        help='Direcotry pointing to PWM matrices'
    )
    parser.add_argument(
                        "--inject_pwm_include_rc",
                        "-irc",
                        action='store_true',
                        default=False,
                        help="Should the reverse complement of PWM injected as well?"
                        )
    parser.add_argument(
        '--inject_pwm_select',
        '-injsel',
        default=[],
        required=False,
        nargs='+',
        help='Selection of DBFs whose PBM will be injected. Needs to match the filename in the PWM directory.'
    )
    parser.add_argument(
        '--numpy',
        '-np',
        default=[],
        required=False,
        nargs='+',
        help='Direcotry pointing to numpy arrays of conv weights and biases saved as conv_weights.npy and conv_bias.npy'
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
        '--deepbind_models',
        '-pdm',
        default=[],
        required=False,
        nargs='+',
        help='Direcotries pointing to pretrained deepbind models to be loaded into the net'
    )
    parser.add_argument(
        '--wavenet_models',
        '-pwm',
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
        default=[0.],
        nargs='+',
        type=float,
        help="""Penalise weight min """)

    parser.add_argument(
        "--weights_max",
        "-wmax",
        default=[0.],
        nargs='+',
        type=float,
        help="""Penalise weight max """)

    parser.add_argument(
        "--u",
        "-u",
        default=[30,10],
        nargs='+',
        type=int,
        help="""Transformation parameter for float to int conversion """)

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
                        "--use_crossvalidation",
                        "-cv",
                        action='store_true',
                        default=False,
                        help="Should training be done in 3fold crossvalidation?"
                        )


    parser.add_argument(
                        "--use_multi_gpu",
                        "-mgpu",
                        action='store_true',
                        default=False,
                        help="Should training be distributed across multiple GPUs?"
                        )
    # Get args
    args = sys.argv[1:]
    args = parser.parse_args(args)

    print('Starting to train model .. ')
    f = filesystem.FileSystem(args.genome_dir ,args.output_base,source_profile= args.profiles,overwrite=False, test_fraction = 0.1, val_fraction=0.2,resume = False)
    tf_preprocessing_params = {'times_median_coverage_max': 3, 'discretize_function': 'float_to_int',
                               'assign_non_covered_bases': None, 'u': args.u[0], 'smooth_signal': True,
                               'discretize_function': 'float_to_int', 'sigma': 5, 'truncate': 3,
                               'smoothing_function': 'gaussian_filter1d', 'x_thresh': 0, 'run_thresh': 10,
                               'normalise_read_counts': 'genome_mean'}
    nuc_preprocessing_params = {'times_median_coverage_max': 3, 'discretize_function': 'float_to_int',
                                'assign_non_covered_bases': None, 'u': args.u[1], 'smooth_signal': True,
                                'discretize_function': 'float_to_int', 'sigma': 5, 'truncate': 3,
                                'smoothing_function': 'gaussian_filter1d', 'x_thresh': 0, 'run_thresh': 50,
                                'normalise_read_counts': 'genome_mean'}

    preprocessing_params = [tf_preprocessing_params, nuc_preprocessing_params]

    run_dataset = runtime_dataset.RuntimeDataset(f)
    run_dataset._set_seed = 32
    run_dataset._shuffle_sequences = True
    run_dataset.save_data = True # this saves the preprocessing params
    run_dataset._include_rc = True
    run_dataset.data_format = 'raw_counts'
    run_dataset.class_weight_cap = [100, 100]
    run_dataset.preprocessing_params = preprocessing_params
    run_dataset._remove_unmapped_training_regions = 0.8
    run_dataset._train_test_split_by_chr = True


    run_dataset.step_size = args.step_size
    run_dataset.fragment_length = args.fragment_length

    run_dataset.load_data()

    # setting min/max weights. too much and the model will only predict highest/lowest class
    weight_logspace_min = args.weights_min *run_dataset.n_output_features
    weight_logspace_max = args.weights_max *run_dataset.n_output_features
    w_arrays = [numpy.ones((_output_bins, _output_bins)) for _output_bins in run_dataset._output_bins]
    zs = [int(numpy.ceil(_output_bins / 2.)) for _output_bins in run_dataset._output_bins]

    for (w_array, z, w_min, w_max) in zip(w_arrays, zs, weight_logspace_min, weight_logspace_max):
        w_array[:z, -z:] = numpy.logspace(w_min, w_max, num=z)
        w_array[-z:, :z] = numpy.logspace(w_max, w_min, num=z)

    main(args,w_arrays)

