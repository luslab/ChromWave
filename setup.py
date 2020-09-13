from setuptools import find_packages
from setuptools import setup

setup(name='ChromWave',
      version='2.0',
      description='ChromWave: Deciphering the DNA-encoded competition between DNA-binding factors with deep neural networks',
      url='https://github.com/luslab/GenomeNet.git',
      author='Aylin Cakiroglu',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
                        # 'biopython==1.76',
                        # 'h5py==2.10.0',
                        # 'hyperas==0.4.1',
                        # 'joblib==0.14.1',
                        # 'keras==2.3.1',
                        # 'matplotlib==2.2.3',
                        # 'pandas==0.24.2',
                        # 'path.py==9.1',
                        # 'pyyaml==5.3.1',
                        # 'roman==1.4',
                        # 'scikit-image==0.14.3',
                        # 'scikit-learn==0.20.3',
                        # 'tensorflow==2.1.0',
                        ],
      package_data={'': ['*.r', '*.R']},
      include_package_data=True)

