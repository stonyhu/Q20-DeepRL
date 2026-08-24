# -*- coding:utf8 -*-

import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.switch_backend("agg")

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats,
                       each_num_episodes,
                       i_episode,
                       save_dir,
                       smoothing_window=10,
                       noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths[:i_episode])
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.savefig("{}/Figure-{}-Length.png".format(save_dir, i_episode / each_num_episodes))
    #if noshow:
    #    plt.close(fig1)
    #else:
    #    plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards[:i_episode]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig("{}/Figure-{}-Reward.png".format(save_dir, i_episode / each_num_episodes))
    #if noshow:
    #    plt.close(fig2)
    #else:
    #    plt.show(fig2)

    # # Plot the win rate over time
    # fig3 = plt.figure(figsize=(10, 5))
    # plt.ylim(0, 1.0)
    # plt.plot(stats.episode_precision[:i_episode])
    # plt.xlabel("Episode")
    # plt.ylabel("Win Rate")
    # plt.title("Win Rate over Time")
    # plt.savefig("{}/Figure-{}-Precision.png".format(save_dir, i_episode / each_num_episodes))


def plot_episode_accuracy(game_times, accuracys, save_dir, i_episode, each_num_episodes):
    plt.figure(figsize=(10, 5))
    plt.ylim(0, 1.0)
    plt.plot(np.array(game_times), np.array(accuracys))
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate over Episode")
    plt.savefig("{}/Figure-{}-Precision.png".format(save_dir, i_episode / each_num_episodes))

def plot_noise_accuracy(noises, accuracys, save_dir, is_predict=False):
    plt.figure(figsize=(10,5))
    #plt.xlim(0, 1.0)
    #plt.ylim(0, 1.0)
    plt.plot(np.array(noises), np.array(accuracys))
    plt.xlabel("Noise")
    plt.ylabel("Win Rate")
    tag_str = "TrainSet" if not is_predict else "TestSet"
    plt.title("Self-playing Win Rate over Noise on {}".format(tag_str))
    plt.savefig("{}/Figure-Noise-Precision-{}.png".format(save_dir, tag_str))


def plot_turn_accuracy(turns, accuracys, save_dir, is_predict=False):
    plt.figure(figsize=(10,5))
    #plt.xlim(0, 1.0)
    #plt.ylim(0, 1.0)
    plt.plot(np.array(turns), np.array(accuracys))
    plt.xlabel("Turn")
    plt.ylabel("Win Rate")
    tag_str = "TrainSet" if not is_predict else "TestSet"
    plt.title("Self-playing Win Rate over Turn on {}".format(tag_str))
    plt.savefig("{}/Figure-Turn-Precision-{}.png".format(save_dir, tag_str))


