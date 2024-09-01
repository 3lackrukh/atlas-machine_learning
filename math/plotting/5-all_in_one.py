#!/usr/bin/env python3
""" Module defines the all_in_one method """
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Plots graphs from tasks 0 - 4 all in one canvas
    """
    # Set up canvas
    fig = plt.figure()
    # Set up subplots
    ax0 = plt.subplot2grid((3, 2), (0, 0))
    ax1 = plt.subplot2grid((3, 2), (0, 1))
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    
    # Task 0
    x0 = np.arange(0, 11)
    y0 = np.arange(0, 11) ** 3
    ax0.set_xlim(0, 10)
    ax0.plot(x0, y0, color='red')

    # Task 1
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180
    ax1.set_xlabel('Height (in)', fontsize='x-small')
    ax1.set_ylabel('Weight (lbs)', fontsize='x-small')
    ax1.set_title('Men\'s Height vs Weight', fontsize='x-small')
    ax1.scatter(x1, y1, color='magenta', marker='o')

    # Task 2
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    ax2.set_xlabel('Time (years)', fontsize='x-small')
    ax2.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax2.set_title('Exponential Decay of C-14', fontsize='x-small')
    ax2.set_xlim(0, 28650)
    ax2.set_yscale('log')
    ax2.plot(x2, y2, '-')

    # Task 3
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)
    ax3.set_xlabel('Time (years)', fontsize='x-small')
    ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
    ax3.set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')
    ax3.set_xlim(0, 20000)
    ax3.set_ylim(0, 1)
    ax3.plot(x3, y31, 'r--', label='C-14')
    ax3.plot(x3, y32, 'g-', label='Ra-226')
    ax3.legend(loc='upper right', fontsize='x-small')

    # Task 4
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    ax4.set_xlabel('Grades', fontsize='x-small')
    ax4.set_ylabel('Number of Students', fontsize='x-small')
    ax4.set_title('Project A', fontsize='x-small')
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 30)
    ax4.set_xticks(np.arange(0, 101, 10))
    ax4.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.7)
    plt.suptitle('All in One')
    plt.show()


if __name__ == "__main__":
    all_in_one()
