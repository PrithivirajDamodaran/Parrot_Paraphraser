import setuptools


setuptools.setup(
    name="parrot",
    version="1.0",
    author="prithiviraj damodaran",
    author_email="d.prithiviraj@gmail.com",
    description="Parrot paraphraser",
    long_description="Parrot paraphraser",
    url="https://github.com/PrithivirajDamodaran/Parrot.git",
    packages=setuptools.find_packages(),
    install_requires=['transformers', 'sentencepiece', 'python-Levenshtein', 'sentence-transformers', 'fuzzywuzzy'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
)

