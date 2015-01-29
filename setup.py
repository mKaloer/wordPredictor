from setuptools import setup

setup(
    name='WordPredictor',
    version='0.1',
    description='Word predictions based on previous n words',
    author='Mads Kalør',
    author_email='mads@kaloer.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7'
    ],
    install_requires=[
        'nltk>=3.0.1',
        'numpy>=1.9.1',
        'scipy>=0.15.1'
    ],

    extras_require = {
        'test': ['nose'],
    },

)
