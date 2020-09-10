from setuptools import setup

setup(
    name='pinterpret',
    version='1.0',
    description='Tools for interpreting machine learning models in python',
    author='Karolina Chalupova',
    author_email='chalupova.karolina@gmail.com',
    license='',
    packages=['pinterpret'],
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
    ],
    zip_safe=False
)