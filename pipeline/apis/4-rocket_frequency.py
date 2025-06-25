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
    # Get all rockets first to create ID to name mapping
    rockets_url = 'https://api.spacexdata.com/v4/rockets'
    rockets_resp = requests.get(rockets_url)
    
    if rockets_resp.status_code != 200:
        return []
    
    rockets = rockets_resp.json()
    rocket_id_to_name = {}
    
    # Create mapping from rocket ID to rocket name
    for rocket in rockets:
        rocket_id = rocket.get('id')
        rocket_name = rocket.get('name')
        if rocket_id and rocket_name:
            rocket_id_to_name[rocket_id] = rocket_name
    
    # Get all launches
    launches_url = 'https://api.spacexdata.com/v4/launches'
    launches_resp = requests.get(launches_url)
    
    if launches_resp.status_code != 200:
        return []
    
    launches = launches_resp.json()
    rocket_counts = {}
    
    # Count launches per rocket using the mapping
    for launch in launches:
        rocket_id = launch.get('rocket')
        if rocket_id and rocket_id in rocket_id_to_name:
            rocket_name = rocket_id_to_name[rocket_id]
            rocket_counts[rocket_name] = rocket_counts.get(rocket_name, 0) + 1
    
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
