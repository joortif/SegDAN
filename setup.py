from setuptools import setup, find_packages
from codecs import open

# For installing PyTorch and Torchvision in Windows
import sys
import subprocess
import pkg_resources

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements("requirements.txt")

def remove_requirements(requirements, remove_keywords):
    new_requirements = []
    for requirement in requirements:
        if not any(keyword in requirement for keyword in remove_keywords):
            new_requirements.append(requirement)
    return new_requirements

# Remove PyTorch and Torchvision from install requirements to handle manually
install_reqs = parse_requirements("requirements.txt")
install_reqs = remove_requirements(install_reqs, ['torch', 'torchvision', 'torchaudio'])

setup(
    name = 'segdan',

    version = '0.1.3',

    author = 'Joaquin Ortiz de Murua Ferrero',
    author_email = 'joortif@unirioja.es',
    maintainer= 'Joaquin Ortiz de Murua Ferrero',
    maintainer_email= 'joortif@unirioja.es',

    url='https://github.com/joortif/SegDAN',

    description = 'AutoML framework for the construction of segmentation models.',

    long_description_content_type = 'text/markdown', 
    long_description = long_description,

    license = 'MIT license',

    packages = find_packages(include=["*"],exclude=["test"]), 
    install_requires = install_reqs,
    python_requires='>=3.9',
    include_package_data=True, 

    classifiers=[

        'Development Status :: 4 - Beta',

        'Programming Language :: Python :: 3.10',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',       

        # Topics
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        "Operating System :: OS Independent",
    ],

    keywords='instance semantic segmentation pytorch huggingface embedding image analysis deep learning active learning computer vision'
)