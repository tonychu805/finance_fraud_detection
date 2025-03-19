from setuptools import setup, find_packages

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open("implementation/requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="fraud_detection",
    version="0.1.0",
    description="Fraud Detection Portfolio Project",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "fraud-detection=implementation.run:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
) 