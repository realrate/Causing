import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causing",
    version="2.0.0",
    author="Dr. Holger Bartel",
    author_email="holger.bartel@realrate.ai",
    description="Causing: CAUSal INterpretation using Graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/realrate/Causing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy~=1.23",
        "pandas~=1.3",
        "scipy~=1.9",
        "sympy~=1.5",
        "networkx~=2.7",
        "pre-commit",  # TODO: move to dev-requirements
    ],
    python_requires=">=3.9",
    setup_requires=["wheel"],
)
