"""
Test of global installation.
"""
from setuptools import setup

setup(name='fuzzyroughrules',
    version='0.1',
    description='Testing installation of Package',
    url='#',
    author='auth',
    author_email='author@email.com',
    license='GPL-3.0',
    python_requires='>=3.10',
    packages=['fuzzyroughrules', 'fuzzyroughrules.federated', 'hhelper', 'quickrules'],
    zip_safe=False
)