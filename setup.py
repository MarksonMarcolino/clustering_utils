from setuptools import setup, find_packages

setup(
    name="clustering_utils",
    version="0.1.0",
    description="Modular clustering benchmark and visualization toolkit",
    author="Markson Marcolino",
    author_email="markson.marcolino@gmail.com",
    url="https://github.com/MarksonMarcolino/clustering_utils",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "umap-learn",
        "hdbscan",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)