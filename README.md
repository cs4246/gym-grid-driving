# gym-grid-driving

The Grid Driving environment is a simple domain featuring simple yet scalable environment for testing various planning and reinforcement learning algorithm.

## Grid Driving

The Grid Driving task involves a "simplified" driving through traffic full of vehicles from one point to the other. At each timestep, the car can move either up, down, or stay on the same lane as it automatically moves forward. The accomplishment of the task would give +10 reward, whereas failure would yield 0 rewards. The sparse nature of the goal and its variability makes the environment suitable as an entry point for initial experimentation on scalability as well as long term planning.

## Usage

Execute the following command to install the package
```
pip install -e .
```

Create the environment this way
```
import gym
import gym_grid_driving

env = gym.make('GridDriving-v0')
```

Example:
```
from gym_grid_driving.envs.grid_driving import ACTIONS

env.reset()
for i in range(12):
    env.render()
    state, reward, done, info = env.step(np.random.choice(ACTIONS))
```