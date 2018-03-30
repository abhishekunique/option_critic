#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning.

"""

import sys
import launcher

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000 # 250000 #TODO: Need to change this
    EPOCHS = 8000
    STEPS_PER_TEST = 130000 # 130000 #TODO: Need to change this
    FRAME_SKIP = 4 #TODO: Need to change this

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop' #TODO: Need to change this
    LEARNING_RATE = .00025 #TODO: Need to change this
    DISCOUNT = .99 #TODO: Need to change this
    RMS_DECAY = .95 #0.95 (Rho)
    RMS_EPSILON = .01 #0.01
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1 #or 0.01 for tuned ddqn
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4 #TODO: Need to change this
    UPDATE_FREQUENCY = 4 #TODO: Need to change this
    REPLAY_MEMORY_SIZE = 1000000  #TODO: Need to change this
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 10000 #30000 for tuned ddqn #TODO: Need to change this
    REPLAY_START_SIZE = 50000 #50000 #TODO: Need to change this
    # RESIZE_METHOD = 'scale' #scale vs crop #TODO: Need to change this
    # RESIZED_WIDTH = 84
    # RESIZED_HEIGHT = 84
    # OFFSET = 18
    # DEATH_ENDS_EPISODE = True
    CAP_REWARD = True
    MAX_START_NULLOPS = 30 #TODO: Need to change this
    OPTIMAL_EPS = 0.05 #0.05 or 0.001 for tuned ddqn
    DOUBLE_Q = False
    MEAN_FRAME = False
    TEMP=1

    TERMINATION_REG = 0.0
    NUM_OPTIONS = 8
    ACTOR_LR = 0.00025 #TODO: Need to change this
    ENTROPY_REG = 0.0 #TODO: Need to change this
    BASELINE=False

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)
