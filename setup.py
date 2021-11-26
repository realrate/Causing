import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causing",
    version="0.1.7",
    author="Dr. Holger Bartel",
    author_email="holger.bartel@realrate.ai",
    description="Causing: CAUSal INterpretation using Graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/realrate/RealRate-Causing",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numdifftools~=0.9.39",
        "numpy~=1.18",
        "pydot~=1.4",
        "pandas~=1.3",
        "scipy~=1.5.4",
        "sympy~=1.5.1",
        "torch~=1.9.1",
        "pre-commit",
    ],
    python_requires=">=3.9",
    setup_requires=["wheel"],
)
