#!/usr/bin/env python3
"""Module to retrieve user location from GitHub API."""
import requests
import sys
from datetime import datetime


def user_location(url):
    """
    Returns the location of a specific user using the GitHub API.

    Parameters:
        url (str): Full GitHub API URL for the user.

    Returns:
        None: Prints the location or error message to stdout.
    """
    response = requests.get(url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_timestamp = response.headers.get('X-Ratelimit-Reset')
        reset_time = datetime.fromtimestamp(int(reset_timestamp))
        wait_time = reset_time - datetime.now()
        minutes = divmod(wait_time.total_seconds(), 60)[0]
        print("Reset in {} min".format(int(minutes)))
    else:
        user_data = response.json()
        print(user_data.get('location', "No location found"))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} <url>".format(sys.argv[0]))
        sys.exit(1)

    user_location(sys.argv[1])
