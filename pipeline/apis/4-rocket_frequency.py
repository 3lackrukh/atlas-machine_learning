#!/usr/bin/env python3
"""Module to retrieve rocket launch frequency from SpaceX API."""
import requests


def rocket_frequency():
    """
    Returns the number of launches per rocket from SpaceX API.

    Returns:
        list: List of tuples (rocket_name, launch_count) sorted by count desc,
              then alphabetically.
    """
    url = 'https://api.spacexdata.com/v4/launches'
    resp = requests.get(url)

    if resp.status_code != 200:
        return []

    launches = resp.json()
    rocket_counts = {}

    # Count launches per rocket
    for launch in launches:
        rocket_id = launch.get('rocket')
        if rocket_id:
            # Get rocket name from rocket ID
            base_url = 'https://api.spacexdata.com/v4/rockets/'
            rocket_url = base_url + rocket_id
            rocket_resp = requests.get(rocket_url)
            if rocket_resp.status_code == 200:
                rocket_data = rocket_resp.json()
                rocket_name = rocket_data.get('name', 'Unknown')
                prev_count = rocket_counts.get(rocket_name, 0)
                rocket_counts[rocket_name] = prev_count + 1

    # Sort by count (descending) then alphabetically
    sorted_rockets = sorted(
        rocket_counts.items(),
        # Sort by count desc, then name asc
        key=lambda x: (-x[1], x[0])
    )

    return sorted_rockets


if __name__ == '__main__':
    rockets = rocket_frequency()
    for rocket_name, count in rockets:
        output = f"{rocket_name}: {count}"
        print(output)
