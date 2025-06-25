#!/usr/bin/env python3
"""Module to retrieve home planets of sentient species from SWAPI."""
import requests


def sentientPlanets():
    """
    Returns a list of names of the home planets of all sentient species.

    Returns:
        list: List of planet names (str) that are home to sentient species.
    """
    url = 'https://swapi-api.hbtn.io/api/species/'
    planets = []
    while url:
        resp = requests.get(url)
        if resp.status_code != 200:
            break
        data = resp.json()
        for species in data.get('results', []):
            if species.get('designation') == 'sentient':
                homeworld = species.get('homeworld')
                if homeworld and homeworld not in planets:
                    # Get planet name from homeworld URL
                    planet_resp = requests.get(homeworld)
                    if planet_resp.status_code == 200:
                        planet_data = planet_resp.json()
                        planet_name = planet_data.get('name')
                        if planet_name:
                            planets.append(planet_name)
        url = data.get('next')
    return planets 