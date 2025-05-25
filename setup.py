from setuptools import find_packages, setup

setup(
    name="envidis",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    description="Environmental Document Intelligence System",
    author="",
    author_email="",
    install_requires=[
        "spacy",
        "argcomplete",
    ],
    entry_points={
        "console_scripts": [
            "envidis=envidis.cli.main:main",
        ],
    },
)
