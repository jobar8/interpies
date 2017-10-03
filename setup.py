#!/usr/bin/env python
# -*- coding: utf 8 -*-
"""
Python installation file.
"""
from setuptools import setup
import re

# convert README file
try:
    from pypandoc import convert
    long_description = convert('README.md', 'rst')
    long_description = long_description.replace("\r","")
except (IOError, ImportError):
    print("Pandoc not found. Long_description conversion failure.")
    # pandoc is not installed, fallback to using raw contents
    import io
    with io.open('README.md', encoding="utf-8") as f:
        long_description = f.read()

# find VERSION
version_file = 'interpies/__init__.py'

with open(version_file, 'r') as f:
    version_string = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                               f.read(), re.M)
    
if version_string is not None:
    VERSION = version_string.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))

CLASSIFIERS = ['Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'Natural Language :: English',
               'License :: OSI Approved :: BSD License',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Topic :: Scientific/Engineering'
               ]

setup(
    name='interpies',
    version=VERSION,
    packages=['interpies'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'rasterio>=1.0',
        'scikit-learn',
        'scikit-image'
    ],
    url='https://github.com/jobar8/interpies',
    license='BSD',
    author='Joseph Barraud',
    author_email='joseph.barraud@geophysicslabs.com',
    description='A collection of functions for reading, displaying, transforming and analyzing geophysical data.',
    long_description=long_description,
    keywords=['geophysics raster gdal gravimetry magnetometry seismic'],
    classifiers=CLASSIFIERS
)
