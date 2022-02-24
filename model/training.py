from .dqn import DQN
from .memory import MemoryReplay
from .hyperparameters import Hyperparameters
import environment.actions as actions
from environment.imageviewer import ImageViewer
from environment.env_wrappers import SmbRender, FrameSkipEnv, RewardWrapper
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
import torch
import csv
from itertools import count
import numpy as np
import random
import copy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def testModel(learning_model, params: Hyperparameters, device, metrics, epoch, i, epsilon):
    # Setup test variables.
    learning_model.eval()
    env, s0, stages = getEnvironment([(8, 1)], params)
    action = 0
    done = False
    r_ep = 0
    x_start = None
    t_start = None

    # Run a test to episode end.
    for i_ep in count():
        # Take an action and observe the result.
        action = learning_model.act(s0, epsilon=0.0, device=device)
        s0, reward, done, info = env.step(action)
        s0 = np.array(s0)
        r_ep += reward
        if not x_start: x_start = info['x_pos']
        if not t_start: t_start = info['time']

        # Update metrics.
        if done:
            metrics['epoch'].append(epoch)
            metrics['i'].append(i),
            metrics['reward'].append(r_ep)
            metrics['x_movement'].append(info['x_pos']-x_start)
            metrics['success'].append(1 if info['flag_get'] else 0)
            metrics['iterations'].append(i_ep+1)
            metrics['time'].append(np.abs(info['time']-t_start))
            break
    
    # Print the results.
    print('E: %7d/%7d, I: %10d, R: % 5d, X: % 6d, S: %1d, T: %3d, Epsilon: %1.5f' % \
          (epoch, params.epochs, i, metrics['reward'][-1], metrics['x_movement'][-1], metrics['success'][-1], metrics['time'][-1], epsilon))


def replayMemories(params: Hyperparameters, learning_model, target_model, sample_memory, loss, optimizer, device):
    # Get a batch of sample sequences.
    s0, a, r, s1, done = sample_memory.sample(params.batch_size)
    s0 = torch.tensor(s0).float().to(device)
    a = torch.tensor(a, dtype=torch.int64).to(device)
    r = torch.tensor(r).to(device)
    s1 = torch.tensor(s1).float().to(device)
    done = torch.tensor(done).int().to(device)

    # Compute the predicted and target Q values.
    q_estimate = learning_model(s0)[np.arange(0, params.batch_size), a]
    with torch.no_grad():
        best_action = torch.argmax(learning_model(s1), dim=1)
        next_q = target_model(s1)[np.arange(0, params.batch_size), best_action]
        q_target = (r + (1 - done.float()) * params.gamma * next_q).float()

    # Calculate loss and backpropagate.
    l = loss(q_estimate, q_target)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    # Return the loss.
    return l.item()


def getEnvironment(stages, params: Hyperparameters, viewer=None):
    if len(stages) == 0:
        stages = params.stages
        random.shuffle(stages)
    stage = stages.pop()
    env = gym_super_mario_bros.SuperMarioBrosEnv(rom_mode='vanilla', lost_levels=False, target=stage)
    env = JoypadSpace(env, actions.action_set[params.actions])
    env = FrameStack(SmbRender(FrameSkipEnv(RewardWrapper(env, clip_reward=params.clip_reward), frame_skip=params.frame_skip, viewer=viewer)), num_stack=params.image_channels)
    render = env.reset()
    return env, np.array(render), stages


def saveMetrics(path, metrics):
    csv_file = csv.writer(open(os.path.join(path, 'metrics.csv'), 'w', newline=''))
    csv_file.writerow([k for k,v in metrics.items()])
    for i in range(len(metrics['epoch'])):
        csv_file.writerow([v[i] for k,v in metrics.items()])


def doCheckpoint(path, epoch, i, params: Hyperparameters, metrics, learning_model, target_model, optimizer, loss):
    if epoch > 0 and epoch % params.metrics_save == 0:
        print("Metrics: e=%d" % (epoch))
        saveMetrics(path, metrics)

    try:
        if epoch > 0 and epoch % params.checkpoint == 0:
            print("Checkpoint: e=%d" % (epoch))
            torch.save({
                'epoch': epoch,
                'i': i,
                'metrics': metrics,
                'learning_model': learning_model.state_dict(),
                'target_model': target_model.state_dict(),
                'optimizer': optimizer,
                'loss': loss,
                'hyperparameters': params.dict()
            }, os.path.join(path, 'checkpoint.pth'))
    except:
        print("Failure to save checkpoint at epoch %d" % (epoch))


def trainModel(model_name, params: Hyperparameters):
    # Initialize epsilon, n_actions, and device.
    epsilon = 1.0
    n_actions = len(actions.action_set[params.actions])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build the models, loss, optimizer, and memory.
    learning_model = DQN(params.image_channels, n_actions).to(device)
    target_model = DQN(params.image_channels, n_actions).to(device)
    target_model = copy.deepcopy(learning_model)
    for p in target_model.parameters():
        p.requires_grad = False
    loss = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(learning_model.parameters(), lr=params.learning_rate, eps=1e-4)
    sample_memory = MemoryReplay(params.mem_size)

    # Initialize training vars.
    path = os.path.join('save', model_name)
    os.makedirs(path, exist_ok=True)
    stages = [] # All stages to choose from, will be populated by getEnvironment().
    i = 0
    epoch = 0
    best_reward = -np.inf
    metrics = {
        'epoch': [],
        'i': [],
        'reward': [],
        'x_movement': [],
        'success': [],
        'iterations': [],
        'time': []
    }

    # Main training loop.
    print('Beginning training')
    while epoch < params.epochs:
        # Set the model to training mode and init the full list of stages to pull from.
        learning_model.train()
        
        # Intiialize actions and environment.
        done = False
        env, s0, stages = getEnvironment(stages, params)

        # Move through the environment to completion (success or failure).
        for i_ep in count():
            # Get the next action and take it, then record the memory.
            action = learning_model.act(s0, epsilon=epsilon, device=device)
            s1, r, done, info = env.step(action)
            s1 = np.array(s1)
            sample_memory.addMemory(s0, action, r, s1, done)
            s0 = s1

            # Train the network on stored memories.
            if i > params.batch_size:
                replayMemories(params, learning_model, target_model, sample_memory, loss, optimizer, device)

            # Update the target network and close out the episode, if done.
            i += 1
            if i % params.target_update == 0:
                print("Target update: i=%d" % (i))
                target_model.load_state_dict(learning_model.state_dict())
            if done:
                break

        # Test the current model, checkpoint the model progress, and print status.
        epoch += 1
        epsilon = max(epsilon*params.explore_decay, params.epsilon_minimum)
        if (epoch % params.test_and_print) == 0:
            testModel(learning_model, params, device, metrics, epoch, i, epsilon)
            if metrics['reward'][-1] > best_reward:
                best_reward = metrics['reward'][-1]
                print("Model: e=%d, r=%d" % (epoch, best_reward))
                torch.save(learning_model.state_dict(), os.path.join(path, 'model.pth'))
                torch.save(params.dict(), os.path.join(path, 'params.pth'))
        doCheckpoint(path, epoch, i, params, metrics, learning_model, target_model, optimizer, loss)

    # Training done!
    path = os.path.join('save', model_name, 'final')
    os.makedirs(path, exist_ok=True)
    torch.save(learning_model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(params.dict(), os.path.join(path, 'params.pth'))
    saveMetrics(path, metrics)
    print('\n***************************')
    print('Model training complete!')


def demo(model_name, world, stage, demo_scale):
    params = Hyperparameters(torch.load(os.path.join('save', model_name, 'params.pth')))
    n_actions = len(actions.action_set[params.actions])
    device = torch.device('cpu')
    model = DQN(params.image_channels, n_actions).to(device)
    model.load_state_dict(torch.load(os.path.join('save', model_name, 'model.pth'), map_location=device))
    model.eval()

    viewer = ImageViewer('Super Mario Bot', 240*demo_scale, 256*demo_scale, monitor_keyboard=True, relevant_keys=None)
    env, render, stages = getEnvironment([(world,stage)], params, viewer=viewer)
    done = False
    action = 0

    # Iterate over the environment until it is over.
    while not done:
        action = model.act(obs=render, epsilon=0.0, device=device)
        state, reward, done, info = env.step(action)
        render = np.array(state)
        if viewer.is_escape_pressed:
            break