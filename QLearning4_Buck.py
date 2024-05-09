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
eng.cd(r'/home/pvm8318/Documents/Reinforcement/reinforcement')
eng.addpath(r'/home/pvm8318/Documents/Reinforcement/reinforcement')
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

# Define websocket function
def websocket ():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    print('Waiting for Simulink to start')
    s.listen(1)
    conn, addr = s.accept()
    return conn

#initialize and hyperparameters
DISCOUNT = 0.99
LEARNING_RATE = 0.1 
epsilon = 0.1
SHOW_EVERY = 2000
num_episodes = 20000
runtime = 10
Vinit = 0
Iinit = 0
duty_step = np.linspace(0, 1, 201)  # 201 possible duty cycle values from 0 to 1

## Buck converter parameters 
Vref = 5
# u = 0
# R = 1.0  # Resistance
# L = 0.1  # Inductance
# C = 1e-3  # Capacitance
# Vin = 12.0  # Input voltage
# Vref = 5.0  # Reference output voltage.0
# # State-space representation of the buck converter
# A = np.array([[0, 1 / C], [-1 / L, -R / L]])
# B = np.array([[0], [1 / L]])
# #steady state calculation
# duty_cycle =Vref/Vin
# Iout = Vref/R
# ILref = Iout/duty_cycle

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
class DoneChecker:
    def __init__(self):
        self.t0 = None
        self.desirable_band = [4.8, 5.2]

    def isdone(self, x, t):
        V = x[0]
        if V >= self.desirable_band[0] and V <= self.desirable_band[1]:
            if self.t0 is None:
                self.t0 = t
            elif t - self.t0 >= 0.5:
                return True
        else:
            self.t0 = None
        
        return False

done_checker = DoneChecker()

# training DQN model
def train_model(batch_size=64):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)

    # Get current Q-values and next Q-values
    current_q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = net(next_states).max(1)[0]
    next_q_values[dones] = 0.0  # Zero Q-values for terminal states
    expected_q_values = rewards + DISCOUNT * next_q_values  # Assuming discount factor gamma = 0.99

    # Loss and optimize
    loss = loss_fn(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Define possible duty cycles

def select_action(state, epsilon):
    """Selects an action using an epsilon-greedy policy."""
    if np.random.random() < epsilon:  # With probability epsilon, select a random action
        action_index = np.random.randint(0, len(duty_step))
        return duty_step[action_index]  # Return the actual duty cycle value
    else:  # Otherwise, select the action with the highest Q-value
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = net(state)
        action_index = q_values.max(1)[1].item()  # Get index of the highest Q-value
        return duty_step[action_index]  # Return the actual duty cycle value
    

def plot_data(time, Vo, duty_cycle,episode,total_reward):
    plt.close()
    fig, ax = plt.subplots()
    plt.title(f'Episode {episode} - Total Reward: {total_reward:.2f}')
    ax2 = ax.twinx()  # Create a twin Axes sharing the x-axis
    ax.plot(time, Vo, color='orangered')
    ax2.plot(time, duty_cycle, color='steelblue')
    ax.set_ylabel('Output Voltage', color='orangered')
    ax2.set_ylabel('reward value', color='steelblue')
    ax.tick_params(axis='y', colors='orangered')
    ax2.tick_params(axis='y', colors='steelblue')
    ax = plt.gca()
    ax.spines['top'].set_color('gray')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('orangered')
    ax.spines['right'].set_color('steelblue')
    plt.savefig(f'plots/episode number {episode}.png')




# training loop
for episode in range(num_episodes):
    try:
        conn.close()
    except Exception as e:
        print(f"Error closing connection: {e}")
    t1 = threading.Thread(target=SimRun)
    t1.start()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future2 = executor.submit(websocket)
        conn = future2.result()
    # Reset the environment and get the initial state
    state = np.array([Vinit, Iinit]) 
    total_reward = 0
    time = 0
    action = select_action(state, epsilon)
    prev_deviation = 0
    prev_u = 0
    Vo = []
    rewardval = []
    t=[]
    iteration = 0
    done = False
    while time < runtime:
        if done:
            break
        send_data(conn, action)
        V, IL, Time = receive_data(conn)
        next_state = np.array([V, IL])
        done = done_checker.isdone(next_state, Time)
        reward, _ = composite_reward(next_state, action, prev_u, prev_deviation)
        prev_deviation = abs(next_state[0] - Vref)
        prev_u = action
        total_reward += reward
        replay_buffer.push(state, action, reward, next_state, False)
        if iteration % 1000 == 0:
            train_model(128)
            action = select_action(next_state, epsilon)

        if iteration % 10 == 0:
            t.append(Time)
            Vo.append(V)
            rewardval.append(reward)
        # if iteration % 10000 == 0:
        #     plot_data(t, Vo, rewardval,episode,total_reward)
        state = next_state
        time = Time
        iteration += 1
    if episode % 20 == 0:
        plot_data(t, Vo, rewardval,episode,total_reward)
    conn.close()
    t1.join()

    # plot_data(t, Vo, rewardval,episode,total_reward)

