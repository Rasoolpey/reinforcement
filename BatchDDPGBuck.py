import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import threading
import concurrent.futures
import matlab.engine
import socket
import struct
import random
import time
from collections import deque
import scipy.io

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Define the range of IP addresses
ips = ["127.0.0.100", "127.0.0.101", "127.0.0.102", "127.0.0.103", "127.0.0.104",
       "127.0.0.105", "127.0.0.106", "127.0.0.107", "127.0.0.108", "127.0.0.109"]

# TCP Connection Parameters
MESSAGE_SIZE = 24  # Each double is 8 bytes + 1 byte for delimiter
DELIMITER = b'\n'
TCP_PORT = 50000
BUFFER_SIZE = MESSAGE_SIZE if MESSAGE_SIZE else 32  # Minimum for two doubles

def send_data(conn, val):
    """Sends a double-precision number."""
    msg = struct.pack('>d', val)
    conn.send(msg)

def receive_data(conn):
    """Receives three double-precision numbers."""
    data = b''
    while len(data) < 24:
        data += conn.recv(24 - len(data))

    val1, val2, Time = struct.unpack('>ddd', data)
    return val1, val2, Time

# Initialize and hyperparameters
DISCOUNT = 0.99
LEARNING_RATE = 0.001
tau = 0.001  # For soft target updates
num_episodes = 20000
runtime = 10
Vinit = 0
Iinit = 0
duty_step = np.linspace(0, 1, 201)  # 201 possible duty cycle values from 0 to 1

## Buck converter parameters 
Vref = 5

# Define Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# Initialize networks
state_dim = 2
action_dim = 1
max_action = 1.0
actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = Actor(state_dim, action_dim, max_action).to(device)
critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)

# Initialize target network weights
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

# Experience Replay
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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

# Define done function
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

# Soft update for target networks
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Training DDPG model
def train_model(batch_size=512):
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.FloatTensor(states).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    actions = torch.FloatTensor(actions).unsqueeze(1).to(device)  # Ensure actions have batch dimension
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

    # Get target Q-values
    next_actions = actor_target(next_states)
    next_q_values = critic_target(next_states, next_actions)
    target_q_values = rewards + DISCOUNT * next_q_values * (1 - dones)

    # Critic loss
    q_values = critic(states, actions)
    critic_loss = nn.MSELoss()(q_values, target_q_values.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor loss
    actor_loss = -critic(states, actor(states)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update target networks
    soft_update(actor_target, actor, tau)
    soft_update(critic_target, critic, tau)

# Select action with exploration noise
def select_action(state, noise_scale=0.1):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = actor(state).cpu().detach().numpy()[0]
    action = np.clip(action + noise_scale * np.random.randn(action_dim), 0.0, 1.0)
    return action.item()  # Return as scalar for single action dimension

def plot_data(time, Vo, duty_cycle, episode, total_reward):
    plt.close()
    fig, ax = plt.subplots()
    plt.title(f'Episode {episode} - Total Reward: {total_reward:.2f}')
    ax2 = ax.twinx()  # Create a twin Axes sharing the x-axis
    ax.plot(time, Vo, color='orangered')
    ax2.plot(time, duty_cycle, color='steelblue')
    ax.set_ylabel('Output Voltage', color='orangered')
    ax2.set_ylabel('Reward Value', color='steelblue')
    ax.tick_params(axis='y', colors='orangered')
    ax2.tick_params(axis='y', colors='steelblue')
    ax = plt.gca()
    ax.spines['top'].set_color('gray')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('orangered')
    ax.spines['right'].set_color('steelblue')
    plt.savefig(f'plots/episode_number_{episode}.png')

# Function to run a single simulation episode
def run_simulations(model, ips):
    print("Starting MATLAB engine and running simulations...")
    eng = matlab.engine.start_matlab()
    eng.cd(r'/home/pvm8318/Documents/Reinforcement/reinforcement')
    eng.addpath(r'/home/pvm8318/Documents/Reinforcement/reinforcement')
    print(f"Running run_simulations with model: {model} and ips: {ips}")
    eng.run_simulations(model, ips, nargout=0)
    eng.quit()
    print("MATLAB simulations completed.")
    
    # Load and print simulation errors
    errors = scipy.io.loadmat('simulation_errors.mat')['errors']
    for i, error in enumerate(errors):
        print(f"Simulation {i + 1} error: {error}")

# Function to handle the websocket connection
def websocket(ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, TCP_PORT))
    print(f'Waiting for Simulink to start on {ip}')
    s.listen(1)
    conn, addr = s.accept()
    print(f'TCP connection established on {ip}')
    return conn

# Function to run a simulation episode
def run_simulation_episode(conn, ip, replay_buffer, episode):
    print(f"Using established connection to {ip}:{TCP_PORT}")

    # Reset the environment and get the initial state
    state = np.array([Vinit, Iinit])
    total_reward = 0
    time = 0
    action = select_action(state)
    prev_deviation = 0
    prev_u = 0
    Vo = []
    rewardval = []
    t = []
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
        replay_buffer.push(state, action, reward, next_state, done)
        action = select_action(next_state)

        if iteration % 10 == 0:
            t.append(Time)
            Vo.append(V)
            rewardval.append(reward)
            print(f"Episode {episode}, IP {ip}, Time {Time}, Reward {reward}, V {V}, IL {IL}, Action {action}")

        state = next_state
        time = Time
        iteration += 1
    
    if episode % 20 == 0:
        plot_data(t, Vo, rewardval, episode, total_reward)
    
    conn.close()
    print(f"Completed episode {episode} for IP {ip}")

# Main execution
if __name__ == "__main__":
    model = 'Buck_Converter'
    episode_per_ip = 20000 // len(ips)  # Number of episodes each IP should handle
    for batch in range(episode_per_ip):
        print(f"Starting batch {batch}")
        
        # Start the MATLAB simulation in a separate thread
        t1 = threading.Thread(target=run_simulations, args=(model, ips))
        t1.start()

        print("Starting websocket connections...")

        # Start the websocket connections concurrently
        connections = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(websocket, ips[i]) for i in range(10)]
            for future in concurrent.futures.as_completed(futures):
                conn = future.result()
                connections.append(conn)

        # Add a small delay to ensure the TCP servers are ready
        time.sleep(2)

        # Use ThreadPoolExecutor to run the simulation episodes concurrently using the established connections
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_simulation_episode, connections[i], ips[i % len(ips)], replay_buffer, batch * len(ips) + i) for i in range(10)]
            concurrent.futures.wait(futures)

        # Wait for the MATLAB simulation thread to complete
        t1.join()

        # Train the model after collecting experiences from 10 simulations
        print(f"Training model after batch {batch}")
        train_model(512)
        print(f"Completed training for batch {batch}")

