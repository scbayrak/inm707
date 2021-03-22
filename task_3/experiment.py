import gym
from model import PPO
from PPO_learning import PpoLearning


# define the environment
env = gym.make('CartPole-v0')
env.reset()

# hyper-parameters for PPO learning
epochs = 4 # number of epochs per learning
no_batches = 2 # number of batches for splitting the timesteps
hidden_dim = 256
gamma = 0.99 # discount factor
gae_lambda = 0.95 # gae smoothing parameter
lr_actor = 0.0003
lr_critic = 0.0005
clip = 0.2 # PPO clipping epsilon parameter
T = 20 # timesteps per each learning

ppo = PPO(env=env, T=T, hidden_dim=hidden_dim, gamma=gamma, 
    gae_lambda=gae_lambda, clip=clip, no_batches=no_batches,
    epochs=epochs, lr_actor=lr_actor, lr_critic=lr_critic)

no_games = 400

training = PpoLearning(no_games, ppo)
training.train()

