import gfootball.env as football_env

# implementing this example for the soccer
# Following this example
# https://www.kaggle.com/denisvodchyts/dqn-tf-agent-with-rule-base-collection-policy
# Support
# https://colab.research.google.com/github/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb
# Commands to execute
# pip install pyvirtualdisplay
# pip install tf-agents
# pip install tensorflow

# from __future__ import absolute_import, division, print_function

import base64
import imageio
# import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf
# print(tf.version.VERSION)
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec, BoundedArraySpec
from tf_agents.utils import common
from tf_agents.networks import q_network


# Learning rate (alpha) and Discount factor (gamma)
from GfootballEnv import GfootballEnv
from GfootballHumanPyPolicy import GfootballHumanPyPolicy
from LearningHelper import LearningHelper

num_iterations = 20000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"} alpha
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# how to render every episode?
env_name = "academy_empty_goal_close"

# Load the environment
env = football_env.create_environment(env_name=env_name, stacked=False, logdir='/tmp/football',
                                      write_goal_dumps=False, write_full_episode_dumps=False, render=False)
env.reset()

print('Observation Spec:')
print(env.observation_space.high)
print(env.observation_space.low)
print('Action Space:')
print(env.action_space)
print('Reward Range:')
print(env.reward_range)

time_step = env.reset()
print('Time step:')
print(time_step)

action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)

# train_py_env = football_env.wrappers.SMMWrapper
# eval_py_env = HaliteWrapper()
# train_py_env.set_opponent_behaviour(behaviour='random_moving', policy_file=None)
# eval_py_env.set_opponent_behaviour(behaviour='random_moving', policy_file=None)
# train_env = tf_py_environment.TFPyEnvironment(train_py_env)
# eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# create test and train environment
print('configuration to put in cfg')
print(env.unwrapped.observation())
cfg = {'ball_rotation',
       'left_team_active',
       'ball_owned_player',
       'right_team_yellow_card',
       'right_team_tired_factor',
       'right_team_roles',
       'ball_owned_team',
       'left_team_tired_factor',
       'ball',
       'left_team',
       'score',
       'game_mode',
       'left_team_yellow_card',
       'steps_left',
       'left_team_roles',
       'right_team_direction',
       'ball_direction',
       'right_team',
       'right_team_active',
       'left_team_direction',
       'designated',
       'active',
       'sticky_actions'
       }

train_env = tf_py_environment.TFPyEnvironment(GfootballEnv(cfg, scenario=env_name))
tf_env_eval = tf_py_environment.TFPyEnvironment(GfootballEnv(cfg, scenario=env_name))


def create_q_network():
    fc_layer_params = (100, 50)
    preprocessing_layers_d = {
        'left_team_yellow_card': tf.keras.layers.Dense(11),
        'left_team_roles': tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=11),
        # tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=10),                        #                           tf.keras.layers.Flatten()]),
        'ball_direction': tf.keras.layers.Dense(3),
        'left_team_tired_factor': tf.keras.layers.Dense(11),
        'left_team_active': tf.keras.layers.Dense(11),
        'right_team_tired_factor': tf.keras.layers.Dense(11),
        'ball': tf.keras.layers.Dense(3),
        'ball_owned_player': tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=12),
        # tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=12),                                                   #tf.keras.layers.Flatten()]),
        'ball_rotation': tf.keras.layers.Dense(1),
        'right_team_active': tf.keras.layers.Dense(11),
        'game_mode': tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=7),
        # tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=7),                                                   #tf.keras.layers.Flatten()]),
        'steps_left': tf.keras.layers.Dense(1),
        'right_team': tf.keras.layers.Flatten(),
        'right_team_yellow_card': tf.keras.layers.Dense(11),
        'left_team': tf.keras.layers.Flatten(),
        'ball_owned_team': tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=3),
        # tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=3),                                                  #tf.keras.layers.Flatten()]),
        'score': tf.keras.layers.Dense(2),
        'right_team_roles': tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=11),
        # tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=12),                                                   #tf.keras.layers.Flatten()]),
        'right_team_direction': tf.keras.layers.Flatten(),
        'left_team_direction': tf.keras.layers.Flatten(),
        'designated': tf.keras.layers.Dense(1),
        'active': tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=12),
        # tf.keras.models.Sequential([tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=12),                                                   #tf.keras.layers.Flatten()]),
        'sticky_actions': tf.keras.layers.Dense(10),
    }
    preprocessing_layers = {}
    for k in cfg:
        preprocessing_layers[k] = preprocessing_layers_d[k]

    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    return q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=fc_layer_params)


# Create DQN Agent
q_net = create_q_network()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# train_step_counter = tf.Variable(0)
global_step = tf.compat.v1.train.get_or_create_global_step()
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step,
    target_update_period=10)

agent.initialize()
agent.train = common.function(agent.train)

# Create policy for data collection
a_spec = train_env.action_spec()
t_spec = train_env.time_step_spec()
mp = GfootballHumanPyPolicy(time_step_spec=t_spec, action_spec=a_spec)

# Agent learning object
magent = LearningHelper(train_env=train_env, test_env=tf_env_eval, agent=agent, global_step=global_step,
                        collect_episodes = 3,
                        eval_interval=5,
                        replay_buffer_capacity=3500,
                        batch_size=500,
                        collect_policy = mp
)
magent.restore_check_point()


# Train one cycle
# I am using here rule based policy for data collection
magent.train_agent(1)
magent.save_policy()

# Evaluate agent
# Evaluation is done on one episode

avg_ret = magent.evaluate_agent(1)
print(avg_ret)

'''

steps = 0
while True:
  obs, rew, done, info = env.step(env.action_space.sample())
  steps += 1
  if steps % 100 == 0:
    print("Step %d Reward: %f" % (steps, rew))
  if done:
    break

print("Steps: %d Reward: %.2f" % (steps, rew))
'''
