import flappy_bird_gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

"""
Artificial Intelligence for Reinforcement Learning that learns how to play FlappyBird

Authors:
- Kamil Rominski
- Artur Jankowski

FlappyBird environment requires python 3.9

Modules to install:
    pip install flappy_bird_gym
    pip install tensorflow=2.5.0
    pip install keras-rl2
"""

def build_model(obs, actions):
    """
    Method that builds model with layers required to learn how to play a game
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(1, obs)))
    model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.summary()
    return model

env = flappy_bird_gym.make("FlappyBird-v0")
"""
Selecting environment for game
"""

obs = env.observation_space.shape[0]
actions = env.action_space.n

model = build_model(obs, actions)
"""
Creating observation space and actions from environment and feeding it to model
"""

def build_agent(model, actions):
    """
    Building agent that will learn based on neural network model
    """
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.5, value_min=.0001, value_test=.0,
                                  nb_steps=6000000)
    memory = SequentialMemory(limit=100000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=500)
    return dqn


dqn = build_agent(model, actions)

dqn.compile(Adam(learning_rate=0.003))
"""dqn.fit(env, nb_steps=5000000, visualize=False, verbose=1)"""
"""
Compiling neural network using Adam model
After that, network will start to learn how to play the game
"""

"""dqn.save_weights("flappy.h5")"""
"""
Saves results of network to file for further reproduction of results
"""

dqn.load_weights("testing.h5")
dqn.test(env, visualize=True, nb_episodes=1)