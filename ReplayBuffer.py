import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._index = 0

    def __len__(self):
        return len(self._storage)

    def append(self, obs, action, reward, termination):
        item = (obs, action, reward, termination)
        if len(self._storage) < self._maxsize:
            self._storage.append(item)
        else:
            self._storage[self._index] = item

        self._index = (self._index + 1) % self._maxsize


    def get_history(self, end_indx, history_size):
        obses = []
        for i in range(history_size):
            obs, _, _, done = self._storage[end_indx-i]
            if i > 1 and done == True:
                break
            obses.append(obs)

        while len(obses) < history_size:
            obses.append(obses[-1])

        obses.reverse()
        return obses

    def sample_random(self, batch_size, history_size):

        double_history_size = history_size + 1

        idxes = [random.randint(history_size, len(self._storage) - 1) for _ in
                 range(batch_size)]
        obses, actions, rewards, obses_t, dones = [], [], [], [], []
        for ind in idxes:
            h_obses = self.get_history(ind, double_history_size)
            obs, action, reward, done = self._storage[ind-1]
            obses.append(np.stack(h_obses[0:-1], axis=2))
            obses_t.append(np.stack(h_obses[1:], axis=2))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        return np.stack(obses, axis=0), np.array(actions), \
            np.array(rewards), np.array(obses_t), np.array(dones)

    def print_contents(self):
        print(self._storage)
    #
    # def _sample_random(self, sample_amount):
    #     rand_indxs = np.random.choice(range(len(self._storage)),
    #                                   sample_amount,
    #                                   replace=False)
    #     # return [self._storage[ind] for ind in rand_indxs]
    #     obs = np.empty([len(rand_indxs), 80, 80, 5])
    #     actions = np.empty([len(rand_indxs)], dtype=np.uint8)
    #     rewards = np.empty([len(rand_indxs)], dtype=np.uint8)
    #     dones = np.empty([len(rand_indxs)], dtype=np.bool)
    #     for i, ind in enumerate(rand_indxs):
    #         obs[i] = self._storage[ind][0]
    #         actions[i] = self._storage[ind][1]
    #         rewards[i] = self._storage[ind][2]
    #         dones[i] = self._storage[ind][3]
    #
    #     return obs, actions, rewards, dones


if __name__ == '__main__':
    rb = ReplayBuffer(4)

    for a in range(8):
        iterr = str(a)
        rb.append('obs' + iterr, 'ac' + iterr, 'r' + iterr, 'term' + iterr)
    rb.print_contents()
