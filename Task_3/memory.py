""" Saves items to memory and retrieves them from memory for optimization.

"""


import numpy as np

class Memory():
    """ PPO algorithm class with the learning algorithm.

        Args:
            no_batches: Number of batches for splitting the items.
            T: Time steps to store in memory for every optimization.
        Attributes:
            no_batches: Number of batches for splitting the items.
            T: Time steps to store in memory for every optimization.
            memory: Dictionary to keep items in memory.

    """

    def __init__(self, no_batches, T):

        self.no_batches = no_batches
        self.T = T
        keys_storage = ["states", "values", "actions",
                "log_probs", "rewards", "dones"]
        self.memory = {k: [] for k in keys_storage}

    def get_batches(self):
        """ Gets batches from the memory for optimization

        Returns:
            self.memory.values(): Batchified items from memory.
            batch_inds: Indices of the batches of items.

        """
        indices = np.arange((self.T), dtype=np.int64)
        batch_inds = np.split(indices, self.no_batches)

        self.memory.update((k, np.array(v))for k, v in self.memory.items())

        return self.memory.values(), batch_inds

    def save_memory(self, states, values, actions, log_probs, rewards, dones):
        """ Saves items to memory

        Args:
            states: States.
            values: State values.
            actions: Actions.
            log_probs: Log-probabilities
            rewards: Rewards
            dones: Done masks

        Returns:
            None

        """
        new_items = {k : v for k, v in locals().items()
                if k not in ["self", "new_items"]}

        for k, v in new_items.items():
            self.memory[k].append(v)

    def reset(self):
        """ Clears memory for before T steps in the environment.

        """
        self.memory = {k : [] for k in self.memory}
