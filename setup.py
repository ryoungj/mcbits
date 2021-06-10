#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
    # 'Click>=7.0',
]


setup(
    name='mcbits',
    version='1.0',
    description="Monte Carlo Bits-Back Coding",
    author="Yangjun Ruan et al.",
    author_email='',
    packages=[
        'mcbits'
    ],
    package_dir={'mcbits': 'mcbits'},
    include_package_data=True,
    license="GNU LESSER GENERAL PUBLIC LICENSE Version 3, 29 June 2007",
    zip_safe=False,
    python_requires='>=2.7',
)