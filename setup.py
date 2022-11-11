# libraries
from setuptools import setup, find_packages


setup(
    # project name
    name='TumorDetector2',
    # current project version
    version='0.1.0',
    # a short description about the project
    description='A neural net which has the capability to identify and segment a tumor on tomograph images.',
    # long description about the project
    long_description=open('README.md', 'r').read(),
    # project licenses
    license='LICENSE.txt',
    # extra project informations
    classifiers=[
        # specifying who is the target public
        'Intended Audience :: Healthcare Industry',
        # specifying licenses about the project
        'License :: Free for non-commercial use',
        # operation systems
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: POSIX :: Linux',
        # specifying all Python version that project supports
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only'
    ],
    # specifying where the main project code is located.
    packages=find_packages(),
    # Python version required
    python_requires='==3.10.6',
    # scripts
    scripts=[
        'run.py'
    ],
    # require packages
    install_requires=[
        'tensorflow==2.10.0',
        'Pillow==9.3.0',
        'tqdm==4.64.1',
        'scikit-learn==1.1.3'
    ]
)