"""
CandleStickBot — Automated Forex Trading Bot
Based on The Candlestick Trading Bible (Munehisa Homma / Steve Nison Methodology)
Version: 0.1.0 (Phase 0 — Foundation)
"""

from setuptools import setup, find_packages

setup(
    name="candlestickbot",
    version="0.1.0",
    description="Automated Forex Trading Bot — Candlestick Bible Methodology",
    author="CandleStickBot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "pyyaml>=6.0.2",
        "pydantic>=2.7.0",
        "sqlalchemy>=2.0.50",
        "alembic>=1.18.0",
        "structlog>=26.1.0",
        "python-dotenv>=1.2.2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-mock>=3.15.0",
            "hypothesis>=6.100.0",
            "pytest-cov>=5.0.0",
        ],
        "data": [
            "pandas>=2.2.0",
            "numpy>=1.26.0",
        ],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
