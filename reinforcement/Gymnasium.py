# Importing Gymnasium for creating the RL environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleBankMarketingEnv(gym.Env):
    """
    Simplified Custom Environment for Bank Marketing Strategy.
    """

    # Metadata for rendering modes
    metadata = {'render.modes': ['console']}

    def __init__(self, data, labels, n_actions=2):
        """
        Initialize the environment with data and labels.

        Args:
        - data: DataFrame containing the features for each state.
        - labels: Series containing the corresponding labels for the data.
        - n_actions: The number of actions available in the environment. Default is 2 (binary actions).
        """
        super(SimpleBankMarketingEnv, self).__init__()

        # Storing the data and labels
        self.data = data
        self.labels = labels
        self.n_actions = n_actions

        # Define action space and observation space
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)

        # Initialize the current step in the data
        self.current_step = 0

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
        - The initial state from the data.
        """
        self.current_step = 0
        return self.data.iloc[self.current_step]

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
        - action: The action chosen by the agent.

        Returns:
        - next_state: The next state after the action.
        - reward: Reward obtained from the action.
        - done: A boolean flag indicating if the episode has ended.
        - {}: Additional info (optional, not used here).
        """
        self.current_step += 1
        done = self.current_step == len(self.data)
        reward = self._calculate_reward(action, self.labels.iloc[self.current_step - 1])

        next_state = None if done else self.data.iloc[self.current_step]

        return next_state, reward, done, {}

    def render(self, mode='console'):
        """
        Render the environment's current state to the console.

        Args:
        - mode: The mode of rendering. Currently only supports 'console'.
        """
        if mode != 'console':
            raise NotImplementedError
        print(f'Step: {self.current_step}, State: {self.data.iloc[self.current_step]}')

    def _calculate_reward(self, action, actual_response):
        """
        Calculate the reward based on the action and actual response.

        Args:
        - action: The action taken by the agent.
        - actual_response: The true label for the current state.

        Returns:
        - A reward of 1 for a correct prediction, -1 for an incorrect prediction.
        """
        if action == actual_response:
            return 1  # Simple reward for a correct prediction
        return -1  # Simple penalty for an incorrect prediction


# Initializing the custom environment with training data
env = SimpleBankMarketingEnv(data=X_train, 
                             labels=y_train, 
                             n_actions=2)

# Example of interacting with the environment
state = env.reset()
done = False
num_steps = 5  # Set the number of steps you want to interact for

for _ in range(num_steps):
    if not done:
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)
        env.render()
    else:
        break
