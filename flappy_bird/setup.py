from setuptools import find_packages, setup

setup(
    name="flappy_bird",
    version="0.1.0",
    install_requires=["gym"],
    packages=find_packages(where=".", exclude=["notebooks"]),
    include_package_data=True,
)
