from setuptools import setup, find_packages

setup(
    name="tdmpc2",
    version="0.1.0",
    description="TDMPC2: Scalable Model-Based RL for Fast Online Adaptation",
    author="TDMPC2 Authors",
    author_email="",  # Add author email if available
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'numpy',
        'gym',
        'pyyaml',
    ],
    include_package_data=True,
    package_data={
        'tdmpc2': ['config.yaml'],
    }
)