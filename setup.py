#!/usr/bin/env python

from setuptools import setup
import subprocess
import sys
import os

# check that python version is 3.5 or above
python_version = sys.version_info
if python_version < (3, 5):
    sys.exit("Python < 3.5 is not supported, aborting setup")
print("Confirmed Python version {}.{}.{} >= 3.5.0".format(*python_version[:3]))

# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


VERSION = '0.2'

setup(name='QPOEstimation',
      description='Estimating QPOs in red noise',
      author='Moritz Huebner',
      author_email='moritz.huebner@monash.edu',
      license="MIT",
      version=VERSION,
      packages=['QPOEstimation', 'QPOEstimation.prior', 'QPOEstimation.model'],
      package_dir={'QPOEstimation': 'QPOEstimation'},
      package_data={'QPOEstimation': ['paper.mplstyle']},
      python_requires='>=3.5',
      install_requires=[
          'bilby',
          'stingray'],
      entry_points={},
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Operating System :: OS Independent"])
