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
from gym_grid_driving.envs.grid_driving import Action
import numpy as np

env.reset()
for i in range(12):
    env.render()
    state, reward, done, info = env.step(np.random.choice(Action))
```

## Configuration

### Example configuration
```
import gym
import gym_grid_driving
from gym_grid_driving.envs.grid_driving import Action, LaneSpec, Point

lanes = [
    LaneSpec(2, [-1, -1]),
    LaneSpec(2, [-2, -1]),
    LaneSpec(3, [-3, -1]),
]

env = gym.make('GridDriving-v0', lanes=lanes, width=8, 
               agent_speed_range=(-1,-1), finish_position=Point(0,1), agent_pos_init=Point(6,1),
               stochasticity=1.0, tensor_state=False, random_seed=13)

env.render()
```

#### Details

* `lanes` accepts a list of `LaneSpec(cars, speed_range)` governing how each lanes should be, with `cars` being integer and `speed_range=[min, max]`, `min` and `max` should also be integer
* `width` specifies the width of the simulator as expected
* `agent_speed_range=[min, max]` 
* Coordinate of the finisih point `finish_position` 
* Coordinate of the agent initial position `agent_pos_init`
* `Action` is an enum containing `[Action.stay, Action.up, Action.down]` which should be self explanatory
* Degree of stochasticity `stochasticity` with `1.0` being fully-stochastic and `0.0` being fully-deterministic
* `tensor_state` whether to output state as 4D tensor `[cars, agent, finish_position, occupancy_trails]`
* `random_seed` to make the environment reproducible

**Notes:** 

* To make the simulator deterministic, one just have to set the `stochasticity=0.0` or `min=max` in the `*speed_range`
* Parking scenario is just a special case where `min=max=0` in the `car_speed_range`

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
  F   -   O   -   -   -   -   -   -   -
  -   -   -   -   -   2   -   -   1   -
  -   -   4   5   3   -   -   -   -   <
========================================
Action.up
========================================
 OF   ~   -   -   -   -   -   -   -   -
  -   -   2   ~   ~   1   ~   ~   <   -
  4   5   3   ~   -   -   -   -   -   -
========================================
Action.up
========================================
  F   -   -   -   -   -   -   <   -   O
  ~   ~   1   ~   ~   -   -   -   -   2
  ~   3   -   -   -   -   -   -   4   5
========================================
Action.stay
========================================
  F   -   -   -   -   -   <   O   ~   -
  ~   ~   -   -   -   -   2   ~   ~   1
  ~   -   -   -   -   -   -   4   5   3
========================================
Action.up
========================================
  F   -   -   -   -   <   O   -   -   -
  -   -   -   2   ~   ~   1   ~   ~   -
  -   -   -   -   -   4   5   3   ~   -
========================================
Action.up
========================================
  F   -   -   -   <   O   -   -   -   -
  2   ~   ~   1   ~   ~   -   -   -   -
  -   -   4   ~   ~   5   3   -   -   -
========================================
Action.up
========================================
  F   -   -   <   O   -   -   -   -   -
  1   ~   ~   -   -   -   -   2   ~   ~
  4   ~   -   5   ~   3   -   -   -   -
========================================
Action.stay
========================================
  F   -   <   O   -   -   -   -   -   -
  -   -   -   -   2   ~   ~   1   ~   ~
  5   ~   ~   -   3   -   -   -   -   4
========================================
Action.stay
========================================
  F   <   O   -   -   -   -   -   -   -
  -   2   ~   ~   1   ~   ~   -   -   -
  -   3   ~   ~   -   -   -   -   4   5
========================================
Action.down
========================================
  F   O   -   -   -   -   -   -   -   -
 ~#   1   ~   ~   -   -   -   -   2   ~
  3   -   -   -   -   -   -   4   5   -
========================================
```

#### Legend

* `<`: Agent
* `F`: Finish point
* `Integer`: Car
* `#`: Crashed agent
* `-`: Empty road
* `~`: Occupancy trails