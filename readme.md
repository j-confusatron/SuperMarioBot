# Super Mario Bot
[Paper](https://github.com/j-confusatron/SuperMarioBot/blob/main/Learning%20SMB.pdf)

Teach an agent to play Super Mario Bros. using DDQN learning!
https://arxiv.org/pdf/1509.06461.pdf

If you'd like to watch a prebuilt agent play level 1-1:
`python smbot.py --mode demo --name 1-1-bot`

## Required Libraries
- Python 3.8
- Pytorch: https://pytorch.org/get-started/locally/
    - Pyvision is required, Pyaudio is not.
    - CUDA is supported for training, but not required.
- C++ Build Tools: NES-Py is a Python wrapper around a C++ NES emulator. Build tools are required to compile the emulator.
    - Windows: https://visualstudio.microsoft.com/visual-cpp-build-tools/, install Visual Studio Build Tools -> Desktop development with C++
    - *nix: Check to see if clang is already installed: `clang --version`, else see https://github.com/Kautenja/nes-py for instructions.
- NES-Py: `pip install nes-py` https://github.com/Kautenja/nes-py (not available via Conda)
- gym-super-mario-bros `pip install gym-super-mario-bros` https://github.com/Kautenja/gym-super-mario-bros (not available via Conda)
- OpenCV `pip install opencv-contrib-python`

## Build Environment
- Windows 11
- Conda 4.10.3
- Python 3.8.12
- CUDA 11.4
- Pytorch 1.10.2

## Hyperparameters
See: /model/hyperparameters.py

## Command Line Args
- --mode        Set the application to train or to run a demo. Values: train, demo. Default is demo.
- --name        Tell the application which agent to train or demo. Default is 'agent'.
- --world       If mode is demo, set the world to load. Default is 1.
- --stage       If mode is demo, set the stage to load. Default is 1.
- --demoscale   If mode is demo, scale the output image by this integer. Scale 1 resolution is 240x256. Default is 4.
- --checkpoint  If exists, will load a checkpoint file from disk to resume training from. If mode is demo, will load the checkpoint file rather than the model file.
- --record      If exists and mode is demo, will record a video of the demo.

## What is going on here?
This application applies a technique called Reinforcement Learning to teach an intelligent agent how to navigate an environment.
In Reinforcement Learning, an agent explores an environment and is rewarded or punished for each action it takes, according to its outcome.
By rewarding and punishing the agent, the agent learns to take the appropriate action at each given step.

Q Learning allows an agent to learn a concrete function to appropriately value every action available at a given state.
Q Learning is guaranteed to converge to the optimal function. However, reaching convergence is exhaustive and not practical
in an environment as varied as Super Mario Bros.

Instead, we substitute the Q function with Q', an estimate of Q, using a neural network. The neural network used here is
a convolutional neural network (cnn), that receives a rendered image from SMB and infers the appropriate action to take.
As we do not have pre-labeled data to train the cnn with, we instead maintain two copies of the cnn: estimated and target.
The estimated model will infer actions for the agent. The target will provide the recursive look-ahead value of Q combined with 
the observed reward for transitioning from state_t=1 to state_t=2, and so on. The estimated model will then be trained according 
to the target + reward. The target model will periodically be updated with the values of estimated, as estimated improves.

`Q_estimate(s_1, a_1) <- Q_est(s_1, a_1)`<br>
`Q_target(s_1, a_1) <- r_1 + gamma * Q_t(s_2, argmax_a(Q_est(s_2, a)))`<br>
`loss <- SmoothL1Loss(Q_estimate, Q_target)`<br>
`AdamW.optimize(Q_est)`