from setuptools import setup

setup(
    name='lp-vertex',
    version="0.1",
    packages=['command'],
    entry_points={
        'console_scripts': ['lpv=command:cli']
    }
)
