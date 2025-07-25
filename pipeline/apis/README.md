# Data Collection - APIs Project

This project demonstrates how to collect and manipulate data from public APIs using Python, focusing on the SWAPI (Star Wars API), GitHub API, and SpaceX API. The scripts showcase best practices for HTTP requests, pagination, rate limiting, and data transformation.

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

Test files are provided for tasks 0 and 1:
```bash
./0-main.py
./1-main.py
```

For other tasks, import and use the functions directly in your code.

## Files

### Core Functionality
- `0-passengers.py` - Returns a list of ships from SWAPI that can hold a given number of passengers
- `0-main.py` - Test file for `0-passengers.py`
- `1-sentience.py` - Returns a list of names of the home planets of all sentient species
- `1-main.py` - Test file for `1-sentience.py`
- `2-user_location.py` - Returns the location of a specific user using the GitHub API
- `3-spacex.py` - Returns the next upcoming SpaceX launch in the required format
- `4-rocket_frequency.py` - Displays the number of launches per rocket using the SpaceX API

## APIs Used

- **SWAPI (Star Wars API)**: `https://swapi-api.hbtn.io/`
- **GitHub API**: `https://api.github.com/`
- **SpaceX API**: `https://api.spacexdata.com/v4/`

## Documentation

- All modules, classes, and functions include clear docstrings following project guidelines.
- Code is PEP8-compliant and commented for clarity.

## Author

- Project by Alexa Orrico, Software Engineer at Holberton School

---

*This project is part of the Holberton School Machine Learning curriculum.* 