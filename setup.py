import setuptools
from setuptools import setup

setup(
    name='coda',
    version='1.0',
    description='CODA: Cascaded Online Detection-Free Alignment with Jump Recovery for Real-Time Score Following',
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
