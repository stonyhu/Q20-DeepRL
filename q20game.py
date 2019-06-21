import numpy as np
from gym import spaces

YES = 2
NO = -2
NOTSURE = 0

action2qid_file = "data/action2qid.question500.txt"
pos_file = "data/pos.people1000.txt"
rightprob_file = "data/rightProb.people1000.question500.txt"
wrongprob_file = "data/wrongProb.people1000.question500.txt"


class Q20Env:

    def __init__(self):
        self.state_pid_dict = None
        self.action_qid_dict = None

        self.posList = None
        self.rightProb = None
        self.wrongProb = None

        self.load_pos(pos_file)
        self.load_state_pid_dict()
        self.load_action_qid_dict(action2qid_file)
        self.load_rightprob(rightprob_file)
        self.load_wrongprob(wrongprob_file)

        # state, action for policy network
        self.state = [p[1] for p in self.posList]
        self.action_space = spaces.Discrete(len(self.action_qid_dict))

        self.threshold_prob = 1.0
        self.threshold_times = 100
        self.max_step_num = 20
        self.turns = 0

        self.guess_idx = np.random.randint(0, len(self.state))

    def load_state_pid_dict(self):
        state_qid_dict = {}
        for i, p in enumerate(self.posList):
            state_qid_dict[i] = p[0]
        self.state_pid_dict = state_qid_dict

    def load_action_qid_dict(self, action2qid_file):
        action_qid_dict = {}
        for line in open(action2qid_file):
            items = line.strip().split("\t")
            if len(items) != 2:
                continue
            action = int(items[0])
            qid = int(items[1])
            if action not in action_qid_dict:
                action_qid_dict[action] = qid
        self.action_qid_dict = action_qid_dict

    def load_pos(self, pos_file):
        posList = []
        for line in open(pos_file):
            items = line.strip().split("\t")
            if len(items) != 2:
                continue
            pid = int(items[0])
            pos = float(items[1])
            posList.append([pid, pos])
        self.posList = posList

    def load_rightprob(self, rightprob_file):
        rightProb = {}
        for line in open(rightprob_file):
            items = line.strip().split("\t")
            if len(items) != len(self.action_qid_dict) + 1:
                continue
            pid = int(items[0])
            if pid not in rightProb:
                rightProb[pid] = [float(x) for x in items[1:]]
        self.rightProb = rightProb

    def load_wrongprob(self, wrongprob_file):
        wrongProb = {}
        for line in open(wrongprob_file):
            items = line.strip().split("\t")
            if len(items) != len(self.action_qid_dict) + 1:
                continue
            pid = int(items[0])
            if pid not in wrongProb:
                wrongProb[pid] = [float(x) for x in items[1:]]
        self.wrongProb = wrongProb

    def choose_answer_v2(self, guess_pid, qid, noise=0.0):
        right_prob = self.rightProb[guess_pid][qid]
        wrong_prob = self.wrongProb[guess_pid][qid]
        if right_prob > 0.5:
            return YES
        elif wrong_prob > 0.5:
            return NO
        elif 1 - right_prob - wrong_prob > 0.5:
            return NOTSURE

        max_prob = max(right_prob, wrong_prob, 1 - right_prob - wrong_prob)
        if max_prob == right_prob:
            return YES if np.random.rand() >= noise else NO
        elif max_prob == wrong_prob:
            return NO if np.random.rand() >= noise else YES
        elif max_prob == 1 - right_prob - wrong_prob:
            return NOTSURE

    def choose_answer(self, guess_pid, qid, noise=0.0):
        right_prob = self.rightProb[guess_pid][qid]
        wrong_prob = self.wrongProb[guess_pid][qid]
        max_prob = max(right_prob, wrong_prob, 1 - right_prob - wrong_prob)
        if max_prob == right_prob:
            return YES if np.random.rand() >= noise else NOTSURE
        elif max_prob == wrong_prob:
            return NO if np.random.rand() >= noise else NOTSURE
        elif max_prob == 1 - right_prob - wrong_prob:
            return NOTSURE

    def step(self, action, estimator_reward, noise=0.0):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state_pid_dict = self.state_pid_dict
        rightProb = self.rightProb
        wrongProb = self.wrongProb
        guess_pid = state_pid_dict[self.guess_idx]
        qid = action
        answer = self.choose_answer(guess_pid, qid, noise=noise)
        for i in range(len(self.state)):
            pid = state_pid_dict[i]
            if answer == YES:
                self.state[i] *= rightProb[pid][qid]
            elif answer == NO:
                self.state[i] *= wrongProb[pid][qid]
            elif answer == NOTSURE:
                self.state[i] *= 1 - rightProb[pid][qid] - wrongProb[pid][qid]

        self.state /= np.sum(self.state)
        # Normalize the state
        sum_pos = np.sum(self.state)
        for i in range(len(self.state)):
            self.state[i] /= sum_pos

        top_list = sorted(self.state, reverse=True)
        times = top_list[0] / top_list[1]
        max_prob = max(self.state)
        max_idx = np.argmax(self.state)

        done = self.turns >= self.max_step_num - 1
        reward = 0.0

        if max_prob == 0:
            reward = self.max_step_num * -1.5
            done = True
        else:
            # When turns < 20, continue
            if self.turns < self.max_step_num - 1:
                # When terminate condition is not satisfied, continue
                if 0 < max_prob < self.threshold_prob:
                    # reward = 0.0
                    # reward = estimator_reward.predict(self.state, action)
                    reward = estimator_reward.predict(self.state, action, self.guess_idx)
                # When terminate condition is satisfied, episode must be over
                elif max_prob >= self.threshold_prob and max_idx == self.guess_idx:
                    reward = self.max_step_num * 1.5
                    done = True
                # When terminate condition is satisfied, episode must be over
                elif max_prob >= self.threshold_prob and max_idx != self.guess_idx:
                    reward = self.max_step_num * -1.5
                    done = True
            # When turns = 20, episode must be over
            else:
                if max_idx == self.guess_idx:
                    reward = self.max_step_num * 1.5
                    done = True
                else:
                    reward = self.max_step_num * -1.5
                    done = True

        self.turns += 1
        return self.state, reward, (self.guess_idx, qid), answer, times, done, {}

    def reset(self, uniform_sample=False):
        # Reset the step counter
        self.turns = 0
        # Reset the state
        if uniform_sample:
            # Uniform random sample
            self.state = np.ones(len(self.posList)) / len(self.posList)
        else:
            # Weighted random sample
            # self.state = softmax([p[1] for p in self.posList])
            self.state = np.array([p[1] for p in self.posList])
            self.state /= np.sum(self.state)
        self.guess_idx = np.random.choice(np.arange(len(self.state)), p=self.state)
        print("Guessed pid: {}".format(self.state_pid_dict[self.guess_idx]))
        return self.state

