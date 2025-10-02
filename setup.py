from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pinn-ceramic-temperature",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Physics-Informed Neural Network for Ceramic Temperature Interpolation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pinn-ceramic-temperature",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="physics-informed neural networks, PINN, temperature interpolation, manufacturing, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pinn-ceramic-temperature/issues",
        "Source": "https://github.com/yourusername/pinn-ceramic-temperature",
        "Documentation": "https://github.com/yourusername/pinn-ceramic-temperature/docs",
    },
)