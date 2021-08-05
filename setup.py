#!/usr/bin/env python3
"""
  setup.py file
"""
import os.path
import re
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

PACKAGES = find_packages(HERE)

with open(os.path.join(HERE, 'README.md')) as readme_file:
  README = readme_file.read()

with open(os.path.join(HERE, 'phenotyper', '__init__.py')) as init_file:
  METADATA = dict(re.findall(r'__([a-z]+)__ = "([^"]+)', init_file.read()))

setup(
  name="phenotyper",
  description=(
    "Pipeline for identifying plant phenotypes from an image"
  ),
  long_description=README,
  version=METADATA['version'],
  author="Ryan Jennings",
  author_email="ryan.jennings1@ucdconnect.ie",
  url="https://github.com/RyanJennings1/Dissertation",
  license="FIXME",
  packages=PACKAGES,
  scripts=[
    'bin/phenotyper',
  ],
  install_requires=[
    'plantcv',
  ],
  classifiers=[
    'Development Status :: 1 - Alpha',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development :: Libraries :: Application Frameworks',
    'Topic :: Software Development :: Libraries :: Python Modules',
  ]
)
