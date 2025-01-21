from setuptools import setup, find_packages


core_dependencies = [
    'numpy',
    'pandas',
    'scipy',
    'pysam',
    'pybedtools',
    'pyBigWig',
]

optional_feature_dependencies = {
    'pytest': ['pytest'],
    'deeptools': ['deeptools'],
    'seaborn': ['seaborn'],
    'matplotlib': ['matplotlib'],
}

all_dependencies = core_dependencies + \
    sum(optional_feature_dependencies.values(), [])

long_description = "Consenrich: a state-estimator for high-resolution, uncertainty-moderated extraction of reproducible numeric signals in noisy multisample HTS data."

setup(
    name='consenrich',
    version='0.0.1b0',
    author='Nolan Holt Hamilton',
    author_email='nolan.hamilton@unc.edu',
    description='Consenrich',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={'consenrich': ['refdata/*']},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords='genomics,atac-seq,chromatin,kalman-filter,epigenetics,data-fusion',
    install_requires=core_dependencies,
    extras_require=optional_feature_dependencies,
    entry_points={
        'console_scripts': [
            'consenrich = consenrich.consenrich:main'
        ]
    },
)
