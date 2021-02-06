#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py.py
# @Author: harry
# @Date  : 2/4/21 8:53 PM
# @Desc  : Train a DQN agent

import time

import numpy as np
import tensorflow as tf

from constants import *
from game_wrapper import GameWrapper
from model import build_q_network
from replay_buffer import ReplayBuffer
from agent import DQNAgent

# Loading and saving information.
# If LOAD_FROM is None, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
# LOAD_FROM = None
LOAD_FROM = 'saved_model'
SAVE_PATH = 'saved_model'
SAVE_REPLAY_BUFFER = True
LOAD_REPLAY_BUFFER = True

WRITE_TENSORBOARD = False
TENSORBOARD_DIR = 'tf_board/'

# If True, use the prioritized experience replay algorithm, instead of regular experience replay
# This is much more computationally expensive, but will also allow for better results. Implementing
# a binary heap, as recommended in the PER paper, would make this less expensive.
USE_PER = False

# How much the replay buffer should sample based on priorities.
# 0 = complete random samples, 1 = completely aligned with priorities
PRIORITY_SCALE = 0.7

TOTAL_FRAMES = 30_000_000  # Total number of frames to train for
EPS_ANNEALING_FRAMES = 700_000
# MAX_EPISODE_LENGTH = 18000  # Maximum length of an episode (in frames)
FRAMES_BETWEEN_EVAL = 10_000  # Number of frames between evaluations
EVAL_LENGTH = 4000  # Number of frames to evaluate for

DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards
MEM_SIZE = 10_000  # The maximum size of the replay buffer
MIN_REPLAY_BUFFER_SIZE = 500  # The minimum size the replay buffer must be before we start to update the agent

UPDATE_FREQ = 4  # Number of actions between gradient descent steps
TARGET_UPDATE_FREQ = 1000  # Number of actions between when the target network is updated

# Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
INPUT_SHAPE = (RESIZED_HEIGHT, RESIZED_WIDTH)
BATCH_SIZE = 32  # Number of samples the agent learns from at once
HISTORY_LENGTH = 4

FRAMES_TO_SKIP = 4
LEARNING_RATE = 0.00025

VISIBLE_TRAINING = True


def train():
    global SAVE_PATH

    # Create environment
    game_wrapper = GameWrapper(
        SCENARIO_CFG_PATH, ACTION_LIST,
        FRAMES_TO_SKIP, HISTORY_LENGTH,
        visible=VISIBLE_TRAINING, is_sync=True
    )

    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # Build main and target networks
    main_dqn = build_q_network(NUM_ACTIONS, LEARNING_RATE, INPUT_SHAPE, HISTORY_LENGTH)
    target_dqn = build_q_network(NUM_ACTIONS, LEARNING_RATE, INPUT_SHAPE, HISTORY_LENGTH)

    replay_buffer = ReplayBuffer(
        MEM_SIZE, INPUT_SHAPE, HISTORY_LENGTH, use_per=USE_PER
    )
    agent = DQNAgent(
        main_dqn, target_dqn, replay_buffer, NUM_ACTIONS,
        INPUT_SHAPE, BATCH_SIZE, HISTORY_LENGTH,
        eps_annealing_frames=EPS_ANNEALING_FRAMES,
        replay_buffer_start_size=MEM_SIZE / 2,
        max_frames=TOTAL_FRAMES,
        use_per=USE_PER,
    )

    # load saved model
    if LOAD_FROM is None:
        frame_number = 0
        rewards = []
        loss_list = []
    else:
        print('Loading from', LOAD_FROM)
        meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

        # Apply information loaded from meta
        frame_number = meta['frame_number']
        rewards = meta['rewards']
        loss_list = meta['loss_list']

        print('Loaded')

    # Main loop
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                epoch_frame = 0
                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    # begin of episode
                    start_time = time.time()
                    game_wrapper.reset()
                    episode_reward_sum = 0
                    terminal = False
                    while not terminal:
                        # Get action
                        action = agent.get_action(frame_number, game_wrapper.state)

                        # Take step
                        processed_frame, reward, terminal = game_wrapper.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward

                        # Add experience to replay memory
                        agent.add_experience(
                            action=action,
                            frame=processed_frame[:, :, 0],
                            reward=reward,
                            terminal=terminal
                        )

                        # perform learning step
                        if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                            loss, _ = agent.learn(
                                BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number,
                                priority_scale=PRIORITY_SCALE
                            )
                            loss_list.append(loss)

                        # Update target network
                        if frame_number % TARGET_UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                            agent.update_target_network()
                    # end of episode

                    print(".", end='', flush=True)
                    rewards.append(episode_reward_sum)

                    # Output the progress every 10 games
                    if len(rewards) % 10 == 0:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                            tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                            writer.flush()

                        print("")
                        print(
                            f'Game number: {str(len(rewards)).zfill(6)}  '
                            f'Frame number: {str(frame_number).zfill(8)}  '
                            f'Average reward: {np.mean(rewards[-10:]):0.1f}  '
                            f'Std: {np.std(rewards[-10:]):0.1f}  '
                            f'Min: {np.min(rewards[-10:]):0.1f}  '
                            f'Max: {np.max(rewards[-10:]):0.1f}  '
                            f'Time taken: {(time.time() - start_time):.1f}s  '
                            f'Epsilon: {agent.calc_epsilon(frame_number):.4f}'
                        )

                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                print("")
                print("Evaluating")
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0
                for _ in range(EVAL_LENGTH):
                    if terminal:
                        print(".", end='', flush=True)
                        game_wrapper.reset()
                        episode_reward_sum = 0
                        terminal = False

                    action = agent.get_action(frame_number, game_wrapper.state, evaluation=True)
                    _, reward, terminal = game_wrapper.step(action)
                    evaluate_frame_number += 1
                    episode_reward_sum += reward

                    # On game-over
                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    # In case the game is longer than the number of frames allowed
                    final_score = episode_reward_sum

                # Print score and write to tensorboard
                print("")
                print('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

                # Save model
                if len(rewards) > 300 and SAVE_PATH is not None:
                    agent.save(
                        SAVE_PATH,
                        frame_number=frame_number, rewards=rewards, loss_list=loss_list
                    )
                    # agent.save(
                    #     f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                    #     rewards=rewards, loss_list=loss_list
                    # )
    except:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(
                SAVE_PATH, save_replay_buffer=SAVE_REPLAY_BUFFER,
                frame_number=frame_number, rewards=rewards, loss_list=loss_list
            )
            print('Saved.')


if __name__ == '__main__':
    train()
