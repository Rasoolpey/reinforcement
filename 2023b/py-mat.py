import matlab.engine
import socket, struct
import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import integrate
import torch.nn.functional as F
import threading
import concurrent.futures
import matplotlib.pyplot as plt

# matlab api connection
eng = matlab.engine.start_matlab()
eng.cd(r'C:\Users\pvm8318\Documents\NeoVim\Reinforcement\2023b')
eng.addpath(r'C:\Users\pvm8318\Documents\NeoVim\Reinforcement\2023b')
def SimRun():
    eng.sim('Buck_Converter.slx')
    return

# Create a new socket and bind to the address and port
def websocket ():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    print('Waiting for Simulink to start')
    s.listen(1)
    conn, addr = s.accept()
    return conn

## TCP Connection
MESSAGE_SIZE = 24
DELIMITER = b'\n'
TCP_IP = '156.62.139.28'
TCP_PORT = 50000
BUFFER_SIZE = MESSAGE_SIZE if MESSAGE_SIZE else 32  # Minimum for two doubles


def send_data(conn, val):
    """Sends two double-precision numbers."""
    # Fixed Size
    msg = struct.pack('>d', val)
    conn.send(msg)

def receive_data(conn):
    """Receives three double-precision numbers."""
    if MESSAGE_SIZE:
        data = conn.recv(MESSAGE_SIZE)
        val1, val2, Time = struct.unpack('>ddd', data)
    else:
        # Delimiter
        val1 = None
        val2 = None
        Time = None
        while True:
            data = conn.recv(BUFFER_SIZE)
            if DELIMITER in data:
                val1_bytes, remaining = data.split(DELIMITER, 1)
                val1 = struct.unpack('>d', val1_bytes)[0]
                if DELIMITER in remaining:
                    val2_bytes, time_bytes = remaining.split(DELIMITER, 1)
                    val2 = struct.unpack('>d', val2_bytes)[0]
                    Time = struct.unpack('>d', time_bytes)[0]
                    break
    return val1, val2, Time


# Buck converter parameters 
Vref = 5
u = 0
R = 1.0  # Resistance
L = 0.1  # Inductance
C = 1e-3  # Capacitance
Vin = 12.0  # Input voltage
Vref = 5.0  # Reference output voltage.0
# State-space representation of the buck converter
A = np.array([[0, 1 / C], [-1 / L, -R / L]])
B = np.array([[0], [1 / L]])
#steady state calculation
duty_cycle =Vref/Vin
Iout = Vref/R
ILref = Iout/duty_cycle

# RL Configuration
input_size = 2
hidden_size = 128
output_size = 1
F = nn.functional
class Agent(nn.Module):
        def __init__(self):
                super(Agent, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.layer2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
                x = F.relu(self.layer1(x))
                x = self.layer2(x)
                return x


agent = Agent()

# RL Agent 
class RandomPolicy(nn.Module):
        def forward(self, state):
                # Random policy: sample a random action
                return torch.rand(1)
        
# Control law
def control_law(x):
        with torch.no_grad():
                action = agent(torch.tensor(x, dtype=torch.float32))
        return action.item()  


# reward calculation
def rewardcal(x, u):
    V = x[0]
    IL = x[1]
    Q = 10*np.eye(2)  # State penalty matrix
    R = 1 
    reward = -np.linalg.norm(x - np.array([Vref, ILref]))**2 
    # reward = -np.linalg.norm(x - np.array([Vref, ILref]))**2 - u**2 * R
    return reward

# isdone function
def isdone(x, t):
    desirable_band = [4.8, 5.2]
    t0 = None
    V = x[0]
    IL = x[1]
    if V >= desirable_band[0] and V <= desirable_band[1]:
        if t0 is None:
            t0 = t
        elif t - t0 >= 0.5:
            return True
    else:
        t0 = None
    return False





# training the agent
num_episodes = 10

# # Run the episodes
for episode in range(num_episodes):
    t1 = threading.Thread(target=SimRun)
    t1.start()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future2 = executor.submit(websocket)
        conn = future2.result()
   
    # interavtive plot mode
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)


    time = []
    Vo = []
    reward_v = []
    duty_cycle = []
    
    t = 0
    u = 0
    while t<20:

        send_data(conn, u)

        val1, val2,t = receive_data(conn)
        x= np.array([val1,val2])
        reward_v.append(rewardcal(x, u))
        time.append(t)
        duty_cycle.append(u)
        u = control_law(x)
        Vo.append(val1)

        # Update the plot with the new data
        ax1.plot(time, Vo, color='orangered')
        ax2.plot(time, duty_cycle, color='steelblue')
        ax3.plot(time, reward_v, color='gold')


        # Set the labels for the left and right y-axes
        ax1.set_ylabel('Output Voltage', color='orangered')
        ax2.set_ylabel('Duty cycle', color='steelblue')
        ax3.set_ylabel('Reward', color='gold')

        # Set the colors for the left and right y-axes
        ax1.tick_params(axis='y', colors='orangered')
        ax2.tick_params(axis='y', colors='steelblue')
        ax3.tick_params(axis='y', colors='gold')

        # Set the color of the y-axis frames
        ax1.spines['left'].set_color('orangered')
        ax2.spines['left'].set_color('steelblue')
        ax3.spines['left'].set_color('gold')

        plt.pause(0.005)  # Pause to allow real-time update

        # print('Duty cycle is:', u)
        # print('time is:', time)
        # print('Output voltage is:', val1)
        # print('reward value is:', rewardcal(x, u))
    # Close the connection
    conn.close()
plt.ioff()  # Disable interactive mode
plt.show()



