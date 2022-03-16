from setuptools import find_packages, setup

setup(name="autograd", packages=find_packages(),
      install_requires=["yapf", "flake8", "isort", "pytest", "numpy"])
