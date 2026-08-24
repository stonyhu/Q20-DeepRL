import os
import sys
import itertools
import matplotlib
import numpy as np
import tensorflow as tf
import collections
from lib import plotting
from q20game import Q20Env
from util import sigmoid
from network import PolicyNetwork, ValueNetwork, RewardNetwork, ObjectAwareRewardNetwork

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

matplotlib.style.use("ggplot")
env = Q20Env()


def reinforce(env, estimator_policy, estimator_value, estimator_reward, num_episodes,
              sess=None, discount_factor=0.99, uniform_sample=False,
              saver=None, model_dir=None, figure_dir=None):
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    total_t = sess.run(tf.train.get_global_step())

    win_count = 0
    game_times = [0]
    win_rates = [0.0]

    fp_r = open(experiment_dir + "/reward.txt", "w")
    fp_r.write("episode\treward\n")
    fp_w = open(experiment_dir + "/winrate.txt", "w")
    fp_w.write("episode\twinrate\n")

    episode_win_rates = np.zeros(num_episodes)

    Transition = collections.namedtuple("Transition",
                                        ["state", "action", "next_state",
                                         "reward", "pq_pair", "done"])

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first state
        state = env.reset(uniform_sample=uniform_sample)

        episode = []
        masked_action = np.ones(env.action_space.n)

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = estimator_policy.predict(state, masked_action)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # Taken actions should be not taken again in the following turns
            masked_action[action] = 0.0

            next_state, reward, pq_pair, answer, times, done, _ = \
                env.step(action, estimator_reward)

            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action, next_state=next_state,
                reward=reward, pq_pair=pq_pair, done=done))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t + 1

            # Print out which step we're on, useful for debugging.
            print("\rTrain Step {} @ Episode {}/{} ({}) {} {}".format(
                t, i_episode + 1, num_episodes,
                stats.episode_rewards[i_episode],
                env.action_qid_dict[action],
                answer))
            sys.stdout.flush()

            if done:
                break

            state = next_state

        loss = None
        masked_action = np.ones(env.action_space.n)
        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # The return after this time step
            total_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode[t:]))
            # Update our value estimator
            estimator_value.update(transition.state, total_return)
            # Calculate baseline/advantage
            baseline_value = estimator_value.predict(transition.state)
            advantage = total_return - baseline_value
            # Update our policy estimator
            loss = estimator_policy.update(
                transition.state, advantage, transition.action, masked_action)
            # Taken actions should be not taken again in the following turns
            masked_action[transition.action] = 0.0

            # Update our reward estimator
            object = transition.pq_pair[0]
            # # Update reward network
            # estimator_reward.update(transition.state, transition.action, sigmoid(total_return))
            # Update object-aware reward network
            estimator_reward.update(transition.state, transition.action, object, sigmoid(total_return))
        total_t += 1

        # Update win rate statistics
        episode_win_rates[i_episode] = win_count * 1.0 / (i_episode + 1)
        if stats.episode_rewards[i_episode] > 0:
            win_count += 1
        if (i_episode + 1) % 10 == 0:
            win_rate = win_count * 1.0 / (i_episode + 1)
            game_times.append(i_episode + 1)
            win_rates.append(win_rate)
        #     fp_w.write(str(i_episode + 1) + "\t" + str(win_rate) + "\n")
        #     fp_w.flush()
        # fp_r.write(str(i_episode + 1) + "\t" + str(stats.episode_rewards[i_episode]) + "\n")
        # fp_r.flush()

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode],
                                  node_name="episode_reward",
                                  tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode],
                                  node_name="episode_length",
                                  tag="episode_length")
        episode_summary.value.add(simple_value=episode_win_rates[i_episode],
                                  node_name="episode_win_rate",
                                  tag="episode_win_rate")
        episode_summary.value.add(simple_value=loss,
                                  node_name="episode_policy_loss",
                                  tag="episode_policy_loss")
        estimator_policy.summary_writer.add_summary(episode_summary, total_t)
        estimator_policy.summary_writer.flush()

        # Save model and plot length & reward curve every m episodes
        if (i_episode + 1) % each_num_episodes == 0:
            # Evaluate model
            evaluate(i_episode, env, estimator_policy, estimator_reward, fp_r, fp_w,
                     evaluated_episodes=2000,
                     uniform_sample=uniform_sample)
        if (i_episode + 1) % 10000 == 0:
            # Save model
            model_path = "{}/episodes_{}_model.ckpt".format(model_dir, i_episode)
            saver.save(sess, model_path)
            print("Model saved in path: {}".format(model_path))
            # Save plot figures
            plotting.plot_episode_stats(stats, 10000, i_episode + 1, figure_dir, smoothing_window=25)
            plotting.plot_episode_accuracy(game_times, win_rates, figure_dir, i_episode + 1, 10000)

    fp_r.close()
    fp_w.close()

    return stats


def evaluate(training_episode,
             env,
             estimator_policy,
             estimator_reward,
             fp_reward,
             fp_winrate,
             evaluated_episodes=2000,
             uniform_sample=False):
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(evaluated_episodes),
        episode_rewards=np.zeros(evaluated_episodes))

    reward_sum = 0
    win_count = 0

    for i_episode in range(evaluated_episodes):
        # Reset the environment and pick the first state
        state = env.reset(uniform_sample=uniform_sample)
        masked_action = np.ones(env.action_space.n)

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = estimator_policy.predict(state, masked_action)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # action = np.argmax(action_probs)
            # Taken actions should be not taken again in the following turns
            masked_action[action] = 0.0

            next_state, reward, pq_pair, answer, times, done, _ = \
                env.step(action, estimator_reward)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t + 1

            # Print out which step we're on, useful for debugging.
            print("\rEvaluate Step {} @ Episode {}/{} ({}) {} {}".format(
                t, i_episode + 1, evaluated_episodes,
                stats.episode_rewards[i_episode],
                env.action_qid_dict[action],
                answer))
            sys.stdout.flush()

            if done:
                break

            state = next_state

        if stats.episode_rewards[i_episode] > 0:
            win_count += 1
            reward_sum += 30
        else:
            reward_sum += -30

    average_reward = reward_sum * 1.0 / evaluated_episodes
    win_rate = win_count * 1.0 / evaluated_episodes
    fp_reward.write(str(training_episode + 1) + "\t" + str(average_reward) + "\n")
    fp_reward.flush()
    fp_winrate.write(str(training_episode + 1) + "\t" + str(win_rate) + "\n")
    fp_winrate.flush()


def self_play(env,
              estimator_policy,
              estimator_reward,
              uniform_sample=False,
              num_episodes=10000,
              noise=0.0,
              turn=20):
    """
    Self play to test the accuracy of trained policy network

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        num_episodes: Number of episodes to run for
        noise:
        save_dir:

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # Set the turns of game
    env.max_step_num = turn
    print("env.max_turns={}".format(env.max_step_num))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    win_count = 0

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first state
        state = env.reset(uniform_sample=uniform_sample)
        masked_action = np.ones(env.action_space.n)

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = estimator_policy.predict(state, masked_action)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # Taken actions should be not taken again in the following turns
            masked_action[action] = 0.0

            next_state, reward, pq_pair, answer, times, done, _ = \
                env.step(action, estimator_reward, noise=noise)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t + 1

            print("\r{} Step {} @ Episode {}/{} ({}) {} {}".format(
                "Selfplay-Turn", t, i_episode + 1, num_episodes,
                stats.episode_rewards[i_episode],
                env.action_qid_dict[action],
                answer))
            sys.stdout.flush()

            if done:
                break

            state = next_state

        # Update win rate statistics
        if stats.episode_rewards[i_episode] > 0:
            win_count += 1

    win_rate = win_count * 1.0 / num_episodes
    return win_rate


def selfplay_with_turn(sess, trained_model_dir, uniform_sample=False):
    print("-----------Load Trained Model------------")
    policy_estimator = PolicyNetwork(input_size=len(env.state), output_size=env.action_space.n)
    policy_estimator.restore(sess=sess,
                             checkpoint_file=os.path.join(trained_model_dir, 'episodes_99999_model.ckpt'))
    reward_estimator = RewardNetwork(input_size=len(env.state), output_size=1)
    reward_estimator.restore(sess=sess,
                             checkpoint_file=os.path.join(trained_model_dir, 'episodes_99999_model.ckpt'))

    print("-----------Simulation with Different Turns------------")
    turn_test_dir = os.path.join(trained_model_dir, "turn-test")
    if not os.path.exists(turn_test_dir):
        os.makedirs(turn_test_dir)
    tag_str = "uniform.winrate.txt" if uniform_sample else "weighted.winrate.txt"
    filename = os.path.join(turn_test_dir, tag_str)
    fp = open(filename, "w")
    fp.write("turn\twinrate\n")

    turns = []
    accuracys = []
    for i in range(20):
        win_rate = self_play(env,
                             policy_estimator,
                             reward_estimator,
                             uniform_sample=uniform_sample,
                             num_episodes=1000,
                             turn=i+1)
        turns.append(i + 1)
        accuracys.append(win_rate)
        fp.write(str(i + 1) + "\t" + str(win_rate) + "\n")
        fp.flush()
    fp.close()


def selfplay_with_noise(sess, trained_model_dir, uniform_sample=False):
    print("-----------Load Trained Model------------")
    policy_estimator = PolicyNetwork(input_size=len(env.state), output_size=env.action_space.n)
    policy_estimator.restore(sess=sess,
                             checkpoint_file=os.path.join(trained_model_dir, 'episodes_99999_model.ckpt'))
    reward_estimator = RewardNetwork(input_size=len(env.state), output_size=1)
    reward_estimator.restore(sess=sess,
                             checkpoint_file=os.path.join(trained_model_dir, 'episodes_99999_model.ckpt'))

    print("-----------Simulation with Different Noises------------")
    noise_test_dir = os.path.join(trained_model_dir, "noise-test")
    if not os.path.exists(noise_test_dir):
        os.makedirs(noise_test_dir)
    tag_str = "uniform.winrate.txt" if uniform_sample else "weighted.winrate.txt"
    filename = os.path.join(noise_test_dir, tag_str)
    fp = open(filename, "w")
    fp.write("noise\twinrate\n")

    noises = []
    accuracys = []
    for i in range(15):
        noise = i * 0.01
        win_rate = self_play(env,
                             policy_estimator,
                             reward_estimator,
                             uniform_sample=uniform_sample,
                             num_episodes=1000,
                             noise=noise)
        # Keep statistics for noise-accuracy curve
        noises.append(noise)
        accuracys.append(win_rate)
        fp.write(str(noise) + "\t" + str(win_rate) + "\n")
        fp.flush()
    fp.close()


max_num_episodes = 100000
each_num_episodes = 250
test_num_episodes = 5000

experiment_dir = "./experiments/people1000-question500-model".format(max_num_episodes)
figure_dir = "{}/figures-train".format(experiment_dir)
model_dir = "{}/models".format(experiment_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


def train():
    tf.reset_default_graph()
    # Create a global step variable
    global_step = tf.Variable(0, name="global_step", trainable=False)

    policy_estimator = PolicyNetwork(input_size=len(env.state), output_size=env.action_space.n,
                                     summaries_dir=experiment_dir)
    value_estimator = ValueNetwork(input_size=len(env.state), output_size=1)
    # Object-aware Reward Network
    reward_estimator = ObjectAwareRewardNetwork(input_size=len(env.state), output_size=1, action_num=env.action_space.n)
    # # Reward Network
    # reward_estimator = RewardNetwork(input_size=len(env.state), output_size=1, action_num=env.action_space.n)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        reinforce(env,
                  policy_estimator,
                  value_estimator,
                  reward_estimator,
                  max_num_episodes,
                  sess,
                  discount_factor=0.99,
                  uniform_sample=False,
                  saver=saver,
                  model_dir=model_dir,
                  figure_dir=figure_dir)


def test(mode="turn"):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        trained_model_dir = "./experiments/uniform-models-episode100000"
        if mode == "turn":
            # selfplay_with_turn(sess, trained_model_dir, uniform_sample=True)
            selfplay_with_turn(sess, trained_model_dir, uniform_sample=False)
        elif mode == "noise":
            # selfplay_with_noise(sess, trained_model_dir, uniform_sample=True)
            selfplay_with_noise(sess, trained_model_dir, uniform_sample=False)


if __name__ == "__main__":
    train()
    # test(mode=sys.argv[1])

# Usage: python train.py

