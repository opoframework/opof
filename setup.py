import os

from setuptools import find_packages, setup

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    long_description = f.read()

setup(
    name="opof",
    version="0.3.0",
    description="Open-source framework for solving the Planner Optimization Problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yiyuan Lee",
    author_email="yiyuan.lee@rice.edu",
    project_urls={
        "Documentation": "https://opof.kavrakilab.org",
        "Source": "https://github.com/opoframework/opof",
    },
    url="https://opof.kavrakilab.org",
    license="BSD-3",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "tqdm",
        "pyyaml",
        "swig",  # Needed for SMAC, otherwise the system package needs to be installed.
        "smac==2.0.0",
        "pypop7",
        "scikit-learn==1.2.0",
        "tensorboard",
        "power_spherical @ git+https://github.com/nicola-decao/power_spherical.git",
    ],
    extras_require={
        "tests": [
            "pytest",
            "pytest-cov",
        ]
    },
    package_data={"opof": ["py.typed"]},
    packages=find_packages(),
    scripts=[
        "scripts/opof-registry",
        "scripts/opof-train",
    ],
    keywords="opof, optimization, machine learning, reinforcement learning, planning, robotics",
)
