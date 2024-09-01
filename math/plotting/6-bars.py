#!/usr/bin/env python3
""" Module defines the bars method """
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """ Plots a stacked bar graph """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))
    plt.bar(np.arange(3), fruit[0], width=0.5, color='red')
    plt.bar(np.arange(3), fruit[1], bottom=fruit[0], width=0.5, color='yellow')
    plt.bar(np.arange(3), fruit[2], bottom=fruit[0] + fruit[1], width=0.5, color="#ff8000")
    plt.bar(np.arange(3), fruit[3], bottom=fruit[0] + fruit[1] + fruit[2], width=0.5, color="#ffe5b4")
    plt.xticks(np.arange(3), ['Farrah', 'Fred', 'Felicia'])
    plt.yticks(np.arange(0, 81, 10))
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend(['apples', 'bananas', 'oranges', 'peaches'])
    plt.show()
