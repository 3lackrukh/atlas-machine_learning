#!/usr/bin/env python3
"""Module to retrieve the next upcoming SpaceX launch in the required
format.
"""
import requests


def upcoming_launch():
    """
    Returns information about the next upcoming SpaceX launch.

    Returns:
        str: Formatted string with launch name, date, rocket, pad, and
        locality.
    """
    api = "https://api.spacexdata.com/"
    launches_url = api + "v4/launches/upcoming/"
    rockets_url = api + "v4/rockets/"
    pads_url = api + "v4/launchpads/"

    resp = requests.get(launches_url)
    if resp.status_code != 200:
        return ""
    launches = resp.json()
    if not launches:
        return ""

    # Sort launches by date_unix to get the soonest
    launch = sorted(
        launches, key=lambda d: d.get('date_unix', float('inf'))
    )[0]

    name = launch.get('name', 'Unknown')
    date_local = launch.get('date_local', '')

    # Get rocket information
    rocket_id = launch.get('rocket')
    rocket_name = "Unknown"
    if rocket_id:
        rocket_resp = requests.get(rockets_url + rocket_id)
        if rocket_resp.status_code == 200:
            rocket_data = rocket_resp.json()
            rocket_name = rocket_data.get('name', 'Unknown')

    # Get launchpad information
    pad_id = launch.get('launchpad')
    pad_name = "Unknown"
    pad_locality = "Unknown"
    if pad_id:
        pad_resp = requests.get(pads_url + pad_id)
        if pad_resp.status_code == 200:
            pad_data = pad_resp.json()
            pad_name = pad_data.get('name', 'Unknown')
            pad_locality = pad_data.get('locality', 'Unknown')

    # Format the output string
    result = "{} ({}) {} - {} ({})".format(
        name, date_local, rocket_name, pad_name, pad_locality
    )
    return result


if __name__ == '__main__':
    print(upcoming_launch())
