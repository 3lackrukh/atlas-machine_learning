# Data Collection - APIs Project

This project demonstrates how to collect and manipulate data from public APIs using Python, focusing on the SWAPI (Star Wars API), GitHub API, SpaceX API, and Coinbase API. The scripts showcase best practices for HTTP requests, pagination, rate limiting, and data transformation.

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
./1-main.py
./2-main.py
./3-main.py
./4-main.py
./5-main.py
```

## Files

### Core Functionality
- `0-passengers.py` - Returns a list of ships from SWAPI that can hold a given number of passengers
- `0-main.py` - Test file for `0-passengers.py`
- `1-sentience.py` - Returns a list of names of the home planets of all sentient species
- `1-main.py` - Test file for `1-sentience.py`
- `2-user_location.py` - Returns the location of a specific user using the GitHub API
- `2-main.py` - Test file for `2-user_location.py`
- `3-spacex.py` - Returns the number of launches for a given rocket using the SpaceX API
- `3-main.py` - Test file for `3-spacex.py`
- `4-company.py` - Returns the number of employees for a given company using the GitHub API
- `4-main.py` - Test file for `4-company.py`
- `5-coinbase.py` - Returns the current price of a given cryptocurrency using the Coinbase API
- `5-main.py` - Test file for `5-coinbase.py`

## APIs Used

- **SWAPI (Star Wars API)**: `https://swapi-api.hbtn.io/`
- **GitHub API**: `https://api.github.com/`
- **SpaceX API**: `https://api.spacexdata.com/v4/`
- **Coinbase API**: `https://api.coinbase.com/v2/`

## Documentation

- All modules, classes, and functions include clear docstrings following project guidelines.
- Code is PEP8-compliant and commented for clarity.

## Author

- Project by Alexa Orrico, Software Engineer at Holberton School

---

*This project is part of the Holberton School Machine Learning curriculum.* 