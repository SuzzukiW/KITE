from setuptools import setup, find_packages

setup(
    name='biokite',
    version='0.1.0',
    description='A package for detecting epistatic interactions using PyTorch',
    author='Xiang Fu',
    author_email='xfu@bu.edu',
    entry_points={
        'console_scripts': [
            'biokite=biokite.cli:main',
        ],
    },
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
    ],
)