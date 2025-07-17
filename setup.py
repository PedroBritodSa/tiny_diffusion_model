from setuptools import setup, find_packages

setup(
    name='tiny_diffusion_model',
    version='0.1',
    description='Educational library for score-based diffusion models and fractal datasets',
    author='Seu Nome',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'tqdm',
        'accelerate',
    ],
    license='MIT',
)
