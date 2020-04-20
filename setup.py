#!/usr/bin/python3

import glob

from setuptools import setup

scripts = glob.glob('bin/*')

description = "Spectroscopic QUasar Extractor and redshift (z) Estimator"

version="1.0"
setup(name="squeze",
    version=version,
    description=description,
    url="https://github.com/iprafols/SQUEzE",
    author="Ignasi Pérez-Ràfols",
    author_email="iprafols@gmail.com",
    packages=['squeze'],
    package_dir = {'': 'py'},
    package_data = {},
    install_requires=['numpy','pandas','argparse','astropy'],
    extras_require={
        'train': ['sklearn'],
        'plot': ['matplotlib']
    }
    test_suite=['squeze.test.test_peakfinder', 'squeze.test.test_training',
                'squeze.test.test_modes',],
    scripts = scripts
    )
