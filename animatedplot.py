import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue  # Import the Queue class

def generate_data(i, queue):
    x = []
    y = []
    z = []
    h = []

    for j in range(i * 10, (i + 1) * 10):
        temp_y = np.random.random()
        x.append(j)
        y.append(temp_y)
        z.append(((j / 10) ** 3 - 10 * j ** 2 + 100 * j - 1000) / 1000)
        h.append((-(j / 10) ** 3 + 7 * j ** 2 + 100 * j - 1000) / 1000)

    queue.put((x, y, z, h))

def plot_data(queue):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_ylim(0, 1)

    while True:
        try:
            x, y, z, h = queue.get(timeout=0.05)
            ax1.plot(x, y, color='orangered')
            ax2.plot(x, z, color='steelblue')
            ax3.plot(x, h, color='gold')
            plt.pause(0.05)
        except queue.Empty:
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    data_queue = Queue()
    process = Process(target=plot_data, args=(data_queue,))
    process.start()

    for i in range(100):
        generate_data(i, data_queue)

    process.join()
