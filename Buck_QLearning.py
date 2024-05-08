import matlab.engine
import socket, struct
import threading
import concurrent.futures
import numpy as np
import torch
import matplotlib.pyplot as plt


print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.version.cuda)


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

def rewardcal(x, u):
    V = x[0]
    IL = x[1]
    Q = 10*np.eye(2)  # State penalty matrix
    R = 1 
    reward = -np.linalg.norm(V - Vref)**2 
    # reward = -np.linalg.norm(x - np.array([Vref, ILref]))**2 - u**2 * R
    return reward


def isdone(x, t):
    # Define the desirable band
    desirable_band = [4.8, 5.2]

    # Initialize the start time and t0
    t0 = None

    V = x[0]
    IL = x[1]
    
    # Check if the state is within the desirable band
    if V >= desirable_band[0] and V <= desirable_band[1]:
        # Check if t0 is None (first time in the band)
        if t0 is None:
            t0 = t
        # Check if the state has been within the desirable band for 0.5 seconds
        elif t - t0 >= 0.5:
            return True
    else:
        # Reset t0 if V gets out of the band
        t0 = None
    
    return False
# Define the Q-table
V_step = 0.1
I_step = 0.1
Imax=20
duty_step = np.linspace(0, 1, 201)
Disctrete_OS_size = [np.int32(Vin/V_step),np.int32(Imax/I_step)]
random_indices = np.random.randint(0, len(duty_step), size=((Disctrete_OS_size) + [duty_step.shape[0]]))
q_table = duty_step[random_indices]


def get_state_index(x):
    V = x[0]
    IL = x[1]
    V_index = np.int32(V/V_step)
    IL_index = np.int32(IL/I_step)
    return V_index, IL_index

def get_q_value(x, u):
    V_index, IL_index = get_state_index(x)
    return q_table[V_index, IL_index, u]

def set_q_value(x, u, value):
    V_index, IL_index = get_state_index(x)
    q_table[V_index, IL_index, u] = value

def get_best_action(x):
    V_index, IL_index = get_state_index(x)
    return np.argmax(q_table[V_index, IL_index])

def get_action(x, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, len(duty_step))
    else:
        return get_best_action(x)
    
def get_state():
    V = eng
    IL = eng
    return np.array([V, IL])

def get_next_state(x, u):
    V = x[0]
    IL = x[1]
    V_next = V + (Vin - R * IL - V) / L
    IL_next = IL + (V - Vin * u) / L
    return np.array([V_next, IL_next])

def get_reward(x, u):
    return rewardcal(x, u)

def update_q_table(x, u, alpha, gamma):
    q_value = get_q_value(x, u)
    next_state = get_next_state(x, u)
    next_q_values = [get_q_value(next_state, next_u) for next_u in range(len(duty_step))]
    next_q_value = np.max(next_q_values)
    reward = get_reward(x, u)
    new_q_value = q_value + alpha * (reward + gamma * next_q_value - q_value)
    set_q_value(x, u, new_q_value)

def run_episode(epsilon, alpha, gamma, conn):
    x = get_state()
    u = get_action(x, epsilon)
    send_data(conn, u)
    V, IL, Time = receive_data(conn)
    x = np.array([V, IL])
    update_q_table(x, u, alpha, gamma)
    return Time

def run_simulation(epsilon, alpha, gamma, conn):
    Time = run_episode(epsilon, alpha, gamma, conn)
    while not isdone(x, Time):
        Time = run_episode(epsilon, alpha, gamma, conn)
    return Time

# Function for plotting
def plot_data(time, Vo, duty_cycle):
    plt.close()
    fig, ax = plt.subplots()
    ax2 = ax.twinx()  # Create a twin Axes sharing the x-axis
    ax.plot(time, Vo, color='orangered')
    ax2.plot(time, duty_cycle, color='steelblue')
    ax.set_ylabel('Output Voltage', color='orangered')
    ax2.set_ylabel('Duty cycle', color='steelblue')
    ax.tick_params(axis='y', colors='orangered')
    ax2.tick_params(axis='y', colors='steelblue')
    ax = plt.gca()
    ax.spines['top'].set_color('gray')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('orangered')
    ax.spines['right'].set_color('steelblue')
    plt.pause(0.1)  # Pause to allow real-time update

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000
epsilon = 0.5
num_episodes = 100
runtime = 10
Vinit = 0
Iinit = 0

for episode in range(num_episodes):
    try:
        conn.close()
    except:
        pass
    t1 = threading.Thread(target=SimRun)
    t1.start()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future2 = executor.submit(websocket)
        conn = future2.result()
    # Reset the environment and get the initial state
    state = np.array([Vinit, Iinit])  # Replace with actual initial state
    total_reward = 0
    time = 0
    u=0
    Vo = []
    rewardval = []
    t=[]
    iteration = 0
    while time < runtime:
        action = get_action(state, epsilon)
        send_data(conn, action)
        V, IL, Time = receive_data(conn)
        next_state = np.array([V, IL])
        reward = get_reward(next_state, action)
        total_reward += reward
        update_q_table(state, action, LEARNING_RATE, DISCOUNT)
        state = next_state
        time = Time
        iteration += 1
        if iteration % 10 == 0:
            t.append(Time)
            Vo.append(V)
            rewardval.append(reward)
        if iteration % 10000 == 0:
            plot_data(t, Vo, rewardval)


    conn.close()
    t1.join()
    print(f"Episode {episode} completed with total reward {total_reward}")
    print(q_table)

