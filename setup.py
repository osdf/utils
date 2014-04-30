#! /usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Christian Osendorfer, osendorf@in.tum.de'


from distutils.core import setup
from setuptools import find_packages


setup(
    name="osdfutils",
    version="pre-0.0",
    author="Christian Osendorfer",
    author_email="osendorf@gmail.com",
    url="http://github.com/osdf/utils/",
    license="BSD license",
    description="Machine Learning Code Snippets and Models",
    packages=find_packages(exclude=['examples', 'docs', 'notebooks']),
    include_package_data=True,
)

