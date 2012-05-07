#! /usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Christian Osendorfer, osendorf@in.tum.de'


from setuptools import setup, find_packages


setup(
    name="utils",
    keywords="Machine Learning Optimization",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
)

