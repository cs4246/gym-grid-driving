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
from gym_grid_driving.envs.grid_driving import Action, Lane

lanes = [
    Lane(2, [-1, -1]),
    Lane(2, [-2, -1]),
    Lane(3, [-3, -1]),
]

env = gym.make('GridDriving-v0', lanes=lanes, width=8, 
               player_speed_range=(-1,0), finish_pos=(1,0), player_pos_init=(1,6))

env.render()
```

#### Details

* `lanes` accepts a list of `Lane(cars, speed_range)` governing how each lanes should be, with `cars` being integer and `speed_range=[min, max]`, `min` and `max` should also be integer
* `width` specifies the width of the simulator as expected
* `player_speed_range=[min, max]` 
* Coordinate of the finisih point `finish_pos` 
* Coordinate of the player initial position `player_pos_init`
* `Action` is an enum containing `[Action.stay, Action.up, Action.down]` which should be self explanatory

**Notes:** 

* To make the simulator deterministic, one just have to set the `min=max` in the `*speed_range`
* Parking scenario is just a special case where `min=max=0` in the `*speed_range`

### Example output
```
================================
  O   -   -   -   -   -   -   O
  F   O   -   O   -   -   <   -
  -   -   O   O   -   O   -   -
================================
```

#### Render every step
```
================================
  O   -   -   -   -   -   -   O
  F   O   -   O   -   -   <   -
  -   -   O   O   -   O   -   -
================================
================================
  -   -   -   -   -   -   O   O
 ~F   O   ~   -   -   -   -   O
  O   O   ~   O   ~   <   -   -
================================
================================
  -   -   -   -   -   O   O   -
 OF   -   -   -   -   -   O   -
  O   -   O   -   <   -   -   O
================================
================================
  -   -   -   -   O   O   -   -
  F   -   -   -   -   O   -   O
  O   ~   -   <   -   O   O   ~
================================
================================
  -   -   -   O   O   -   -   -
  F   -   -   O   ~   O   ~   -
  -   -   -   <   O   O   -   O
================================
================================
  -   -   O   O   -   -   -   -
  F   O  ~#   O   ~   -   -   -
  -   O   O   ~   O   ~   ~   -
================================
```

#### Legend

* `<`: Player
* `F`: Finish point
* `O`: Car
* `#`: Crashed player
* `-`: Empty road
* `~`: Occupancy trail left behind by the car movement (because car can move multiple grid at each timestep)