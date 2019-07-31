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
import numpy as np

state = env.reset()
for i in range(12):
    env.render()
    state, reward, done, info = env.step(np.random.choice(env.actions))
```

## Configuration

### Example configuration
```
import gym
import gym_grid_driving
from gym_grid_driving.envs.grid_driving import LaneSpec, Point

lanes = [
    LaneSpec(2, [-1, -1]),
    LaneSpec(2, [-2, -1]),
    LaneSpec(3, [-3, -1]),
]

env = gym.make('GridDriving-v0', lanes=lanes, width=8, 
               agent_speed_range=(-3,-1), finish_position=Point(0,1), agent_pos_init=Point(6,1),
               stochasticity=1.0, tensor_state=False, flicker_rate=0.5, random_seed=13)

actions = env.actions

env.render()
```

#### Details

* `lanes` accepts a list of `LaneSpec(cars, speed_range)` governing how each lanes should be, with `cars` being integer and `speed_range=[min, max]`, `min` and `max` should also be integer
* `width` specifies the width of the simulator as expected
* `agent_speed_range=[min, max]` is the agent speed range which affects the available actions
* Coordinate of the finisih point `finish_position` 
* Coordinate of the agent initial position `agent_pos_init`
* `env.actions` is an enum containing all available actions which would change depending on the `agent_speed_range`
* `env.action_space` is the OpenAI gym action space that can be sampled, with the definitions defined in `env.actions`
* Degree of stochasticity `stochasticity` with `1.0` being fully-stochastic and `0.0` being fully-deterministic
* `tensor_state` whether to output state as 3D tensor `[channel, height, width]` with `channel=[cars, agent, finish_position, occupancy_trails]`
* `random_seed` to make the environment reproducible
* `flicker_rate` specifies how often the observation will not be available (blackout)

**Notes:** 

* To make the simulator deterministic, one just have to set the `stochasticity=0.0` or `min=max` in the car `speed_range`
* Parking scenario is just a special case where `min=max=0` in the car `speed_range`

### Example output (default configuration)
```
========================================
  F   -   -   -   -   O   -   -   -   -
  -   -   2   1   -   -   -   -   -   -
  -   -   4   3   -   -   -   5   -   <
========================================
```

#### Render every step
```
Start
========================================
  F   -   -   O   -   -   -   -   -   -
  -   -   -   -   -   2   -   -   -   1
  -   -   3   5   -   -   -   4   -   <
========================================
down
========================================
  F   -   O   -   -   -   -   -   -   -
  -   -   2   ~   ~   -   1   ~   ~   -
  -   3   5   -   -   4   ~   -   <   -
========================================
up
========================================
 OF   ~   -   -   -   -   -   -   -   -
  ~   ~   -   1   ~   ~   -   <   -   2
  3   5   -   -   4   -   -   -   -   -
========================================
forward[-3]
========================================
  F   -   -   -   -   -   -   -   -   O
  1   ~   ~   -   <   -   2   ~   ~   -
  ~   -   4   ~   -   -   -   -   3   5
========================================
down
========================================
  F   -   -   -   -   -   -   O   ~   -
  -   -   -   2   ~   ~   -   1   ~   ~
  ~   ~   -   <   -   -   3   5   ~   4
========================================
down
========================================
  F   -   -   -   -   -   O   -   -   -
  2   ~   ~   -   1   ~   ~   -   -   -
  -   -   <   3   ~   5   ~   -   4   -
========================================
up
========================================
  F   -   -   -   O   ~   -   -   -   -
  -  1#   ~   ~   -   -   -   2   ~   ~
  -   -   3   5   ~   4   ~   ~   -   -
========================================
```

#### Legend

* `<`: Agent
* `F`: Finish point
* `Integer`: Car
* `#`: Crashed agent
* `-`: Empty road
* `~`: Occupancy trails