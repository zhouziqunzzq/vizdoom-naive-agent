#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : game_wrapper.py
# @Author: harry
# @Date  : 2/4/21 7:15 PM
# @Desc  : A wrapper class for VizDoom game

import time
import vizdoom as vzd
import numpy as np

from utils import process_frame
from constants import *


class GameWrapper:
    def __init__(
            self,
            scenario_cfg_path=SCENARIO_CFG_PATH, action_list=ACTION_LIST,
            frames_to_skip=4,
            history_length=4,
            visible=False,
            is_sync=True,
    ):
        game = vzd.DoomGame()
        game.load_config(scenario_cfg_path)
        game.set_window_visible(visible)
        if is_sync:
            game.set_mode(vzd.Mode.PLAYER)
        else:
            game.set_mode(vzd.Mode.ASYNC_PLAYER)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.init()
        self.env = game
        self.action_list = action_list
        self.frames_to_skip = frames_to_skip
        self.history_length = history_length

        self.state = None
        self.frame = None

    def reset(self):
        """
        Resets the environment
        """
        self.env.new_episode()
        init_state = self.env.get_state()
        self.frame = init_state.screen_buffer

        # For the initial state, we stack the first frame history_length times
        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=-1)

    def step(self, action):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
        Returns:
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
        """
        self.env.set_action(self.action_list[action])
        reward = self.env.get_last_reward()
        new_frame = self.frame
        terminal = self.env.is_episode_finished()
        for _ in range(self.frames_to_skip):
            self.env.advance_action()
            terminal = self.env.is_episode_finished()
            if terminal:
                break
            else:
                reward = self.env.get_last_reward()
                state = self.env.get_state()
                new_frame = state.screen_buffer

        # reward = self.env.make_action(self.action_list[action], self.frames_to_skip)
        # state = self.env.get_state()
        # new_frame = state.screen_buffer
        # terminal = self.env.is_episode_finished()

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=-1)

        return processed_frame, reward, terminal


def test_game_wrapper():
    g = GameWrapper(visible=True, is_sync=True)
    g.reset()
    print("state shape: ", g.state.shape)

    for e in range(5):
        g.reset()
        for i in range(100):
            s, r, t = g.step(np.random.randint(0, len(g.action_list)))
            print(s.shape, r, t)
            if t:
                break
        print("End of episode {}".format(e))


if __name__ == '__main__':
    test_game_wrapper()
