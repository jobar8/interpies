#!/usr/bin/env python
# -*- coding: utf 8 -*-
"""
Python installation file.
"""
from setuptools import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()


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
    version='0.1.0',
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
    long_description=read_md('README.md'),
    keywords=['geophysics raster gdal gravimetry magnetometry seismic']
)
