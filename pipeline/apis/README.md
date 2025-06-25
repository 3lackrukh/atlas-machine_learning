# Data Collection - APIs Project

This project demonstrates how to collect and manipulate data from public APIs using Python, focusing on the SWAPI (Star Wars API) and other external services. The scripts showcase best practices for HTTP requests, pagination, rate limiting, and data transformation.

## Description

Implementation of key API data collection and transformation techniques, including:
- Making HTTP GET requests
- Handling pagination and rate limits
- Fetching and processing JSON resources
- Manipulating and filtering data from external APIs

## Requirements

- Python 3.9
- Ubuntu 20.04 LTS
- requests (Python package)
- pycodestyle (version 2.11.1)

## Installation

Install the required Python package:
```bash
pip install --user requests
```

## Usage

All scripts are executable. Run them directly from the command line:
```bash
./0-main.py
```

## Files

### Core Functionality
- `0-passengers.py` - Returns a list of ships from SWAPI that can hold a given number of passengers
- `0-main.py` - Test file for `0-passengers.py`

## Documentation

- All modules, classes, and functions include clear docstrings following project guidelines.
- Code is PEP8-compliant and commented for clarity.

## Author

- Project by Alexa Orrico, Software Engineer at Holberton School

---

*This project is part of the Holberton School Machine Learning curriculum.* 