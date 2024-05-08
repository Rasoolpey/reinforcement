import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matlab.engine
import socket, struct
import threading
import concurrent.futures
from collections import deque
import random

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

## matlab api connection
eng = matlab.engine.start_matlab()
eng.cd(r'/home/pvm8318/Documents/Reinforcement/2023b')
eng.addpath(r'/home/pvm8318/Documents/Reinforcement/2023b')
def SimRun():
    eng.sim('Buck_Converter.slx')
    return

## TCP Connection
MESSAGE_SIZE = 24 # each doubles 8 bytes + 1 byte for delimiter
DELIMITER = b'\n'
TCP_IP = '156.62.80.83'
TCP_PORT = 50000
BUFFER_SIZE = MESSAGE_SIZE if MESSAGE_SIZE else 32  # Minimum for two doubles


def send_data(conn, val):
    """Sends two double-precision numbers."""
    # Fixed Size
    msg = struct.pack('>d', val)
    conn.send(msg)

def receive_data(conn):
    """Receives three double-precision numbers."""
    data = b''
    while len(data) < 24:
        data += conn.recv(24 - len(data))

    val1, val2, Time = struct.unpack('>ddd', data)
    return val1, val2, Time
    

## Buck converter parameters 
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
def websocket ():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    print('Waiting for Simulink to start')
    s.listen(1)
    conn, addr = s.accept()
    return conn


# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # Input layer (Voltage and Current)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, len(duty_step))  # Output layer (Actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Network and Optimizer
net = DQN().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Experience Replay
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.temp_buffer = []

    def push(self, state, action, reward, next_state, done, update_interval=1000):
        self.temp_buffer.append((state, action, reward, next_state, done))
        
        # Process temp_buffer periodically or based on other conditions
        if len(self.temp_buffer) >= update_interval or done:
            cumulative_reward = 0
            # Traverse the temporary buffer backwards to accumulate rewards
            for i in reversed(range(len(self.temp_buffer))):
                state, action, r, next_state, d = self.temp_buffer[i]
                cumulative_reward += r  # Accumulate reward
                # Update the stored reward to be the cumulative reward
                self.temp_buffer[i] = (state, action, cumulative_reward, next_state, d)
            
            # Move all processed experiences to the main buffer
            self.buffer.extend(self.temp_buffer)
            self.temp_buffer.clear()  # Clear the temporary buffer

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


replay_buffer = ReplayBuffer()

# Define reward functions
def reward_stability(x):
    V = x[0]
    Vref = 5.0  # Target voltage
    deviation = V - Vref
    penalty = deviation**2  # Squared penalty
    return -penalty  # Negative because we want to minimize penalty

def reward_efficiency(u, prev_u):
    control_effort = (u - prev_u)**2  # Penalize large changes between consecutive actions
    return -0.01 * control_effort  # Scale down to balance against stability reward

def reward_convergence(current_deviation, prev_deviation):
    improvement = prev_deviation - current_deviation
    return improvement  # Positive reward for improvement

def composite_reward(x, u, prev_u, prev_deviation):
    current_deviation = abs(x[0] - 5.0)  # Assuming x[0] is V and target is 5.0
    
    # Calculate individual components
    stability = reward_stability(x)
    efficiency = reward_efficiency(u, prev_u)
    convergence = reward_convergence(current_deviation, prev_deviation)
    
    # Weigh components (these weights can be adjusted based on specific system requirements)
    weight_stability = 1.0
    weight_efficiency = 0.1
    weight_convergence = 0.5

    # Calculate composite reward
    total_reward = (weight_stability * stability +
                    weight_efficiency * efficiency +
                    weight_convergence * convergence)
    
    return total_reward, current_deviation


# defining is done function
class EnvironmentMonitor:
    def __init__(self):
        self.t0 = None

    def isdone(self, x, t):
        desirable_band = [4.8, 5.2]
        V = x[0]

        if V >= desirable_band[0] and V <= desirable_band[1]:
            if self.t0 is None:
                self.t0 = t
            elif t - self.t0 >= 0.5:
                return True
        else:
            self.t0 = None
        
        return False

def train_model(batch_size=32):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Get current Q-values and next Q-values
    current_q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = net(next_states).max(1)[0]
    next_q_values[dones] = 0.0  # Zero Q-values for terminal states
    expected_q_values = rewards + 0.99 * next_q_values  # Assuming discount factor gamma = 0.99

    # Loss and optimize
    loss = loss_fn(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



# Simulation or training loop
state = initial_state  # Define your initial state based on your system
action = select_action(state)  # Select initial action using your strategy

for step in range(num_simulation_steps):
    if step % 1000 == 0:  # Change action every 1000 steps
        action = select_action(state)
    
    # Simulate action in the environment and obtain next state and reward
    next_state, reward, done = environment.step(action)  # You'll need to define this method
    
    # Store in replay buffer
    replay_buffer.push(state, action, reward, next_state, done)

    # Train model periodically
    if step % 20 == 0:  # Adjust training frequency as needed
        train_model(64)
    
    state = next_state
    
    if done:
        break

print("Training completed.")
