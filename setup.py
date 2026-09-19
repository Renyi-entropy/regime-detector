from setuptools import find_packages, setup

setup(
    name="regime-detector",
    version="0.1.0",
    description="Causal, unsupervised regime detector -- witnesses which phase a time series is in, without predicting what's next",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples"]),
    install_requires=["numpy"],
    python_requires=">=3.8",
    license="MIT",
)
