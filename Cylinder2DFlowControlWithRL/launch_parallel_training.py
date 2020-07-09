import argparse
import os
import sys
import csv
import socket
import numpy as np

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from simulation_base.env import resume_env, nb_actuations
from RemoteEnvironmentClient import RemoteEnvironmentClient


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

if host == 'None':
    host = socket.gethostname()

example_environment = resume_env(plot=False, dump_CL=100, dump_debug=1, dump_vtu=50)

use_best_model = True

environments = []
for crrt_simu in range(number_servers):
    environments.append(RemoteEnvironmentClient(
        example_environment, verbose=0, port=ports_start + crrt_simu, host=host,
        timing_print=(crrt_simu == 0)     # Only print time info for env_0
    ))

network = [dict(type='retrieve', tensors = ['obs','prev_obs_1','prev_obs_2','prev_obs_3']), dict(type='dense', size=512), dict(type='dense', size=512)]

agent = Agent.create(
    # Agent + Environment
    agent='ppo',  # Agent specification
    environment=example_environment,  # Environment object
    max_episode_timesteps=nb_actuations,  # Upper bound for number of timesteps (action steps) per episode
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,  # Policy NN specification
    # Optimization
    batch_size=number_servers,  # Number of episodes per update batch #20 default
    learning_rate=1e-4,  # Optimizer learning rate
    subsampling_fraction=0.2,  # Fraction of batch timesteps to subsample
    optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, # The epsilon of the ppo CLI objective
    estimate_terminal=True,  # Whether to estimate the value of terminal states
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    critic_network=network,  # Critic NN specification
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,  # To discourage policy from being too 'certain'
    # TensorFlow etc
    parallel_interactions=number_servers,  # Maximum number of parallel interactions to support
    saver=dict(directory=os.path.join(os.getcwd(), 'saver_data')),  # TensorFlow saver configuration for periodic implicit saving
    summarizer=dict(directory=os.path.join(os.getcwd(), 'saver_data','summary'), labels=['graph', 'entropy', 'kl-divergence', 'losses', 'rewards'])
)

runner = Runner(
    agent=agent,
    max_episode_timesteps=nb_actuations,
    evaluation=False,    # whether to use last of parallel environments as evaluation (deterministic)
    environments=environments,  # List of environment objects from which we gather experience
    remote = "multiprocessing"
)

runner.run(
    num_episodes=1200,
    sync_episodes=True,  # Whether to synchronize parallel environment execution on episode-level
)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episodes,
    ar=np.mean(runner.episode_rewards[-100:]))
)

name = "returns_tf.csv"
if (not os.path.exists("saved_models")):
    os.mkdir("saved_models")

# If continuing previous training - append returns
if (os.path.exists("saved_models/" + name)):
    prev_eps = np.genfromtxt("saved_models/" + name, delimiter=';',skip_header=1)
    offset = int(prev_eps[-1,0])
    print(offset)
    with open("saved_models/" + name, "a") as csv_file:
        spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
        for ep in range(len(runner.episode_rewards)):
            spam_writer.writerow([offset+ep+1, runner.episode_rewards[ep]])
# If strating training from zero - write returns
elif (not os.path.exists("saved_models/" + name)):
    with open("saved_models/" + name, "w") as csv_file:
        spam_writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
        spam_writer.writerow(["Episode", "Return"])
        for ep in range(len(runner.episode_rewards)):
            spam_writer.writerow([ep+1, runner.episode_rewards[ep]])

runner.close()
agent.save()
agent.close()

for environment in environments:
    environment.close()

print("Agent and Runner closed -- Learning complete -- End of script")
