from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="indian-econsultation-analysis",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Sentiment analysis and summarization for Indian e-consultation comments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/indian-econsultation-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "mypy>=1.6.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "econsult-train-sentiment=scripts.train_sentiment:main",
            "econsult-train-summarization=scripts.train_summarization:main",
            "econsult-serve=scripts.serve_api:main",
        ],
    },
)