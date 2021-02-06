## Basic
```python
RESIZED_HEIGHT, RESIZED_WIDTH = 30, 40

# If True, use the prioritized experience replay algorithm, instead of regular experience replay
# This is much more computationally expensive, but will also allow for better results. Implementing
# a binary heap, as recommended in the PER paper, would make this less expensive.
USE_PER = False

# How much the replay buffer should sample based on priorities.
# 0 = complete random samples, 1 = completely aligned with priorities
PRIORITY_SCALE = 0.7

TOTAL_FRAMES = 30_000_000  # Total number of frames to train for
EPS_ANNEALING_FRAMES = 100_000
# MAX_EPISODE_LENGTH = 18000  # Maximum length of an episode (in frames)
FRAMES_BETWEEN_EVAL = 2_000  # Number of frames between evaluations
EVAL_LENGTH = 500  # Number of frames to evaluate for

DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards
MEM_SIZE = 10_000  # The maximum size of the replay buffer
MIN_REPLAY_BUFFER_SIZE = 500  # The minimum size the replay buffer must be before we start to update the agent

UPDATE_FREQ = 4  # Number of actions between gradient descent steps
TARGET_UPDATE_FREQ = 1000  # Number of actions between when the target network is updated

# Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
INPUT_SHAPE = (RESIZED_HEIGHT, RESIZED_WIDTH)
BATCH_SIZE = 32  # Number of samples the agent learns from at once
HISTORY_LENGTH = 1

FRAMES_TO_SKIP = 12
LEARNING_RATE = 0.00025

VISIBLE_TRAINING = True

# model.py
x = Conv2D(8, (6, 6), strides=4, kernel_initializer=VarianceScaling(scale=2.),
           activation='relu', use_bias=False)(x)
x = Conv2D(8, (3, 3), strides=2, kernel_initializer=VarianceScaling(scale=2.),
           activation='relu', use_bias=False)(x)
```

## Deadly corridor
```python
RESIZED_HEIGHT, RESIZED_WIDTH = 60, 80
# If True, use the prioritized experience replay algorithm, instead of regular experience replay
# This is much more computationally expensive, but will also allow for better results. Implementing
# a binary heap, as recommended in the PER paper, would make this less expensive.
USE_PER = False

# How much the replay buffer should sample based on priorities.
# 0 = complete random samples, 1 = completely aligned with priorities
PRIORITY_SCALE = 0.7

TOTAL_FRAMES = 30_000_000  # Total number of frames to train for
EPS_ANNEALING_FRAMES = 700_000
# MAX_EPISODE_LENGTH = 18000  # Maximum length of an episode (in frames)
FRAMES_BETWEEN_EVAL = 10_000  # Number of frames between evaluations
EVAL_LENGTH = 4000  # Number of frames to evaluate for

DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards
MEM_SIZE = 10_000  # The maximum size of the replay buffer
MIN_REPLAY_BUFFER_SIZE = 500  # The minimum size the replay buffer must be before we start to update the agent

UPDATE_FREQ = 4  # Number of actions between gradient descent steps
TARGET_UPDATE_FREQ = 1000  # Number of actions between when the target network is updated

# Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
INPUT_SHAPE = (RESIZED_HEIGHT, RESIZED_WIDTH)
BATCH_SIZE = 32  # Number of samples the agent learns from at once
HISTORY_LENGTH = 4

FRAMES_TO_SKIP = 4
LEARNING_RATE = 0.00025

VISIBLE_TRAINING = True

# model.py
x = Conv2D(8, (6, 6), strides=4, kernel_initializer=VarianceScaling(scale=2.),
           activation='relu', use_bias=False)(x)
x = Conv2D(16, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.),
           activation='relu', use_bias=False)(x)
x = Conv2D(16, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.),
           activation='relu', use_bias=False)(x)
x = Conv2D(32, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.),
           activation='relu', use_bias=False)(x)
```
