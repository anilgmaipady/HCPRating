#!/usr/bin/env python3
"""
Setup script for RD Rating System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rd-rating-system",
    version="1.0.0",
    author="RD Rating System Team",
    author_email="team@rd-rating.com",
    description="A comprehensive system for rating Registered Dietitians based on telehealth session transcripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/rd-rating-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Healthcare",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rd-rating=src.cli:main",
            "rd-rating-server=src.deployment.start_server:main",
            "rd-rating-train=src.training.fine_tune:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="dietitian rating telehealth ai ml healthcare",
    project_urls={
        "Bug Reports": "https://github.com/your-org/rd-rating-system/issues",
        "Source": "https://github.com/your-org/rd-rating-system",
        "Documentation": "https://github.com/your-org/rd-rating-system/docs",
    },
) 