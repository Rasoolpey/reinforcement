import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Enable interactive mode
plt.ion()

# Create an empty plot
fig, ax = plt.subplots()
ax2 = ax.twinx()  # Create a twin Axes sharing the x-axis
ax.set_ylim(0, 1)
# Set the limits for the left y-axis plot


x = []
y = []
z = []

for i in range(100):
    temp_y = np.random.random()
    x.append(i)
    y.append(temp_y)
    z.append(((i/10)**3-10*i**2+100*i-1000)/1000)

    # Update the plot with the new data
    ax1.plot(x, y, color='orangered')
    ax2.plot(x, z, color='steelblue')

    # Set the labels for the left and right y-axes
    ax1.set_ylabel('Left Y-axis', color='orangered')
    ax2.set_ylabel('Right Y-axis', color='steelblue')

    # Set the colors for the left and right y-axes
    ax1.tick_params(axis='y', colors='orangered')
    ax2.tick_params(axis='y', colors='steelblue')

    # Set the color of the y-axis frames
    ax = plt.gca()
    ax.spines['top'].set_color('gray')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('orangered')
    ax.spines['right'].set_color('steelblue')

    plt.pause(0.05)  # Pause to allow real-time update

# Show the final plot
plt.ioff()  # Disable interactive mode
plt.show()