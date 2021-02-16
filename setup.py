import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="causing", # Replace with your own username
    version="0.0.5",
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
        "numdifftools==0.9.39",
        "numpy==1.18.1",
        "pydot==1.4.1",
        "pandas==1.1.5",
        "scipy==1.5.4",
        "svglib==1.0.0",
        "sympy==1.5.1",
        "torch==1.5.0",
        "torchvision==0.6.0",
        "matplotlib==3.1.3"
    ],
    python_requires='>=3.6',

)
