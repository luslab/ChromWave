import datetime
import os
import shutil

class FileSystem:
    """
    Small system level object that can contain information about the runtime, where necessary it can be used to ferry runtime properties around
    """

    _source_data_path = None
    _output_path = None

    # File system behaviours
    _overwrite = False
    _resume = False
    _is_profiling = False

    # Parameters used when training
    _is_training = False
    _val_fraction = 0.3
    _test_fraction = 0.1

    def __init__(self, source_genome_data, output_path,  source_profile=None,overwrite=False, resume=False, test_fraction=None, val_fraction=None):
        assert os.path.exists(source_genome_data),' FATAL ERROR: could not find source data'
        self._source_data_path = os.path.abspath(source_genome_data)

        # Create our own dir so we are in control of it
        if os.path.dirname(output_path) != "chromwave_output":
            output_path = os.path.join(output_path, "chromwave_output")
        self._output_path = os.path.abspath(output_path)
        self._checkpoint_path = os.path.join(self._output_path, 'checkpoints')

        if not isinstance(source_profile, list):
            source_profile = [source_profile]
        self._profile_data_path = [os.path.abspath(s) for s in source_profile if s is not None]


        #If test and val fractions are given we are in training mode
        if test_fraction and val_fraction:
            print("Evaluation and validation data will be generated from the training data with a fractional weighting of: " + str(test_fraction) + " and " + str(val_fraction)  )
            self._is_training = True
            self._test_fraction = test_fraction
            self._validation_fraction = val_fraction
        else:

            self._is_training = False

        self._overwrite = overwrite
        self._resume = resume
        self._build()
        
    def _build(self):
        # Lets check the output
        if not os.path.isdir(self._output_path):
            os.makedirs(self._output_path)

        # We get errors if there is an existing file here...
        if self._is_training:
            if os.path.isdir(self._checkpoint_path):
                shutil.rmtree(self._checkpoint_path)

            os.makedirs(self._checkpoint_path)

        elif os.listdir(self._output_path):
            # Options allow for resuming (cpu save), overwrite (hd space save) and move (history) - Default
            if self._overwrite:
                shutil.rmtree(self._output_path)
                os.makedirs(self._output_path)

            elif not self._resume:
                os.rename(self._output_path, self._output_path + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                os.makedirs(self._output_path)

    def get_genome_directory(self):
        return self._source_data_path

    def get_output_directory(self):
        return self._output_path

    def get_checkpoint_directory(self):
        return self._checkpoint_path

    def get_profile_directory(self):
        return self._profile_data_path

    def is_training(self):
        return self._is_training

    def get_test_fraction(self):
        return self._test_fraction

    def get_validation_fraction(self):
        return self._validation_fraction

    def is_attempting_to_resume(self):
        return self._resume

    def is_profiling(self):
        return self._is_profiling
