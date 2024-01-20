from setuptools import setup, find_packages

setup(
    name="represent",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "represent=src.__main__:cli",
        ],
    },
    install_requires=[
        "fire",
        "uvicorn"
    ],
    author="Martin Christoph Frank",
    author_email="martinchristophfrank@gmail.com",
    description="represent text in different ways.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
