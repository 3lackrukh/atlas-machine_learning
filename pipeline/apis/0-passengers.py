#!/usr/bin/env python3
"""
Module retrieves ships from SWAPI that can hold a given number of passengers.
"""
import requests


def availableShips(passengerCount):
    """
    Returns a list of ships that can hold at least `passengerCount` passengers.

    Parameters:
        passengerCount (int): Minimum number of passengers the ship must hold.

    Returns:
        list: List of ship names (str) that meet the criteria.
    """
    url = 'https://swapi-api.hbtn.io/api/starships/'
    ships = []
    while url:
        resp = requests.get(url)
        if resp.status_code != 200:
            break
        data = resp.json()
        for ship in data.get('results', []):
            passengers = ship.get('passengers', '0').replace(',', '') \
                .replace('unknown', '0')
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                continue
        url = data.get('next')
    return ships
