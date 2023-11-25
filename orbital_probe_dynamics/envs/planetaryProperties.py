""" A series of constants depiciting render and planetary properties for the solar system. 
"""
import numpy as np

renderColourProperties = [
    (253, 184, 19),  # SUN
    (177, 173, 173),  # Mecury
    (255, 69, 0),  # Venus
    (0, 0, 160),  # Earth
    (156, 46, 53),  # Mars
    (255, 140, 0),  # Jupiter
    (222, 184, 135),  # Saturn
    (198, 211, 227),  # Uranus
    (63, 84, 186),  # Neptune
    (221, 196, 175),  # Pluto
    (255, 255, 255),  # Player
]

renderSizeProperties = 2 * np.array([14, 4, 5, 6, 5, 10, 8, 7, 8, 4, 2])

orbitalProperties = [
    {
        "m": 3.3011e23 * 5.02785e-31,
        "r": 2439.7 * 6.68459e-9,
        "a": 0.307499,
        "e": 0.205630,
        "omega": 48.331 / 180 * np.pi,
        "baseTheta": 3.646587180648046,
    },  # Mercury
    {
        "m": 4.8675e24 * 5.02785e-31,
        "r": 6051.8 * 6.68459e-9,
        "a": 0.723332,
        "e": 0.006772,
        "omega": 54.884 / 180 * np.pi,
        "baseTheta": 3.3103089705183937,
    },  # Venus
    {
        "m": 5.972168e24 * 5.02785e-31,
        "r": 6371.0 * 6.68459e-9,
        "a": 1.0003,
        "e": 0.0167086,
        "omega": 114.20783 / 180 * np.pi,
        "baseTheta": 5.434714385763949,
    },  # Earth
    {
        "m": 6.4171e23 * 5.02785e-31,
        "r": 3389.5 * 6.68459e-9,
        "a": 1.52368055,
        "e": 0.0934,
        "omega": 286.5 / 180 * np.pi,
        "baseTheta": 4.616635830069312,
    },  # Mars
    {
        "m": 1.8982e27 * 5.02785e-31,
        "r": 69911 * 6.68459e-9,
        "a": 5.2038,
        "e": 0.0489,
        "omega": 273.867 / 180 * np.pi,
        "baseTheta": 2.9944708647763765,
    },  # Jupiter
    {
        "m": 5.6834e26 * 5.02785e-31,
        "r": 58232 * 6.68459e-9,
        "a": 9.5826,
        "e": 0.0565,
        "omega": 339.392 / 180 * np.pi,
        "baseTheta": 5.131453744114779,
    },  # Saturn
    {
        "m": 8.6810e25 * 5.02785e-31,
        "r": 25362 * 6.68459e-9,
        "a": 19.19126,
        "e": 0.04717,
        "omega": 96.998857 / 180 * np.pi,
        "baseTheta": 3.4220412230683985,
    },  # Uranus
    {
        "m": 1.02413e26 * 5.02785e-31,
        "r": 24622 * 6.68459e-9,
        "a": 30.07,
        "e": 0.008678,
        "omega": 273.187 / 180 * np.pi,
        "baseTheta": 2.748364305379124,
    },  # Neptune
    {
        "m": 1.303e22 * 5.02785e-31,
        "r": 1188.3 * 6.68459e-9,
        "a": 39.482,
        "e": 0.2488,
        "omega": 113.834 / 180 * np.pi,
        "baseTheta": 0.0530998561211129,
    },  # Pluto
]


spaceShipThrustProperties = {
    "availableDeltaV": 5000 / 29800,  # Convert m/s to rebound units
    "thrustPerDT": 1 / 100,  # The amount of thrust the spaceship uses per burn
}
