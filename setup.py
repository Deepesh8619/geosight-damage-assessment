from setuptools import setup, find_packages

setup(
    name="geosight-damage-assessment",
    version="1.0.0",
    description="Satellite-based post-disaster building damage assessment using deep learning",
    author="Deepesh Sharma",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "rasterio>=1.3.0",
        "geopandas>=0.13.0",
        "xarray>=2023.1.0",
        "dask[complete]>=2023.5.0",
        "torch>=2.0.0",
        "segmentation-models-pytorch>=0.3.3",
        "albumentations>=1.3.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "loguru>=0.7.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "geosight-assess=scripts.run_assessment:main",
            "geosight-train=scripts.train_segmentation:main",
        ]
    },
)
