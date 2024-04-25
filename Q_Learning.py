import gym
import numpy as np
from gym import Wrapper
from gym.wrappers import TimeLimit


# class MaxEpisodeStepsWrapper(Wrapper):
#     def __init__(self, env, max_steps):
#         super().__init__(env)
#         self.max_steps = max_steps
#         self.current_step = 0

#     def reset(self, **kwargs):
#         self.current_step = 0
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         self.current_step += 1
#         if self.current_step >= self.max_steps:
#             self.env._max_episode_steps = self.current_step
#             return self.env.step(action)
#         else:
#             return self.env.step(action)

env = gym.make("MountainCar-v0", render_mode="human")
# env = MaxEpisodeStepsWrapper(env, 200)
# env = wrappers.Monitor(env, "./gym-results", force=True)
env = TimeLimit(env, max_episode_steps=200)


# state = env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 2000


Disctrete_OS_size = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / Disctrete_OS_size

# print(discrete_os_win_size)
# print(len(env.observation_space.high))

q_table = np.random.uniform(low=-2, high=0, size=(Disctrete_OS_size + [env.action_space.n]))

print(env.action_space.n)
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))  # Convert to integer array


for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        # print(episode)
        render = True
    else:
        render = False
    descrete_state = get_discrete_state(env.reset()[0])
    done = False
    # print(f"Episode {episode}")
   
    step_counter = 0
    while not done:
        action = np.argmax(q_table[descrete_state])
        obs, reward, terminated, truncated, info = env.step(action)  # Unpack the returned values correctly
        done = terminated or truncated
        new_descrete_state = get_discrete_state(obs)
        # print(step_counter)
        # print(f"render: {render}")
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_descrete_state])
            current_q = q_table[descrete_state + (action,)]
            
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[descrete_state + (action,)] = new_q
        elif obs[0] >= env.goal_position:
            q_table[descrete_state + (action,)] = 0
            print(f"Goal reached at episode {episode}")
        descrete_state = new_descrete_state
        step_counter += 1
        if done:
            break
env.close()

# descrete_state = get_discrete_state(env.reset()[0])
# done = False
# while not done:
#     action = np.argmax(q_table[descrete_state])
#     obs, reward, terminated, truncated, info = env.step(action)  # Unpack the returned values correctly
    
#     new_descrete_state = get_discrete_state(obs)
    
#     env.render()
#     if not done:
#         max_future_q = np.max(q_table[new_descrete_state])
#         current_q = q_table[descrete_state + (action,)]
        
#         new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
#         q_table[descrete_state + (action,)] = new_q
#     elif obs[0] >= env.goal_position:
#         q_table[descrete_state + (action,)] = 0
#         print(f"Goal reached at episode {episode}")
#     descrete_state = new_descrete_state
# env.close()
