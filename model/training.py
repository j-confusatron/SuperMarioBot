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
import cv2
import csv
from itertools import count
import numpy as np
import random
import copy
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def testModel(learning_model, params: Hyperparameters, device, metrics, epoch, i, epsilon):
    # Setup test variables.
    learning_model.eval()
    env, s0, stages = getEnvironment([params.test_stage], params)
    action = 0
    done = False
    r_ep = 0
    x_start = None
    t_start = None
    i_m = int(epoch/params.test_and_print)

    # Run a test to episode end.
    for i_ep in count():
        # Take an action and observe the result.
        action = learning_model.act(torch.tensor(s0, requires_grad=False).to(device), epsilon=0.0)
        s0, reward, done, info = env.step(action)
        s0 = np.array(s0)
        r_ep += reward
        if not x_start: x_start = info['x_pos']
        if not t_start: t_start = info['time']

        # Update metrics.
        if done:
            metrics['epoch'][i_m] = epoch
            metrics['i'][i_m] = i
            metrics['reward'][i_m] = r_ep
            metrics['x_movement'][i_m] = info['x_pos']-x_start
            metrics['success'][i_m] = 1 if info['flag_get'] else 0
            metrics['iterations'][i_m] = i_ep+1
            metrics['time'][i_m] = np.abs(info['time']-t_start)
            if i_m >= 10:
                metrics['mean'][i_m] = np.mean(metrics['reward'][i_m-10:i_m])
            break
    
    # Print the results.
    print('E: %7d/%7d, I: %10d, R: % 5d, MR: % 5.1f, X: % 6d, S: %1d, T: %3d, Epsilon: %1.5f' % \
          (epoch, params.epochs, i, metrics['reward'][i_m], metrics['mean'][i_m], \
           metrics['x_movement'][i_m], metrics['success'][i_m], metrics['time'][i_m], epsilon))


def replayMemories(params: Hyperparameters, learning_model, target_model, sample_memory, loss, optimizer, device):
    # Get a batch of sample sequences.
    s0, a, r, s1, done = sample_memory.sample(params.batch_size, device)

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


def saveMetrics(path, metrics, epoch, params: Hyperparameters):
    csv_file = csv.writer(open(os.path.join(path, 'metrics.csv'), 'w', newline=''))
    csv_file.writerow([k for k,v in metrics.items()])
    for i in range(int(epoch/params.test_and_print)+1):
        csv_file.writerow([v[i] for k,v in metrics.items()])


def doCheckpoint(path, epoch, i, params: Hyperparameters, metrics, learning_model, target_model, optimizer, loss, epsilon, candidate_name):
    if epoch > 0 and epoch % params.metrics_save == 0:
        print("Metrics: e=%d" % (epoch))
        saveMetrics(path, metrics, epoch, params)

    try:
        if epoch > 0 and epoch % params.checkpoint == 0:
            print("Checkpoint: e=%d" % (epoch))
            torch.save({
                'epoch': epoch,
                'i': i,
                'metrics': metrics,
                'epsilon': epsilon,
                'learning_model': learning_model.state_dict(),
                'target_model': target_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hyperparameters': params.dict(),
                'candidate_name': candidate_name
            }, os.path.join(path, 'checkpoint.pth'))
    except:
        print("Failure to save checkpoint at epoch %d" % (epoch))
    
def loadCheckpoint(model_name, device, learning_model, target_model, optimizer, sample_memory):
    print('Loading checkpoint')
    chk = torch.load(os.path.join('save', model_name, 'checkpoint.pth'), map_location=device)

    # Load the models.
    learning_model.load_state_dict(chk['learning_model'])
    target_model.load_state_dict(chk['target_model'])
    learning_model = learning_model.to(device)
    target_model = target_model.to(device)
    for p in target_model.parameters():
        p.requires_grad = False
    target_model.eval()
    optimizer.load_state_dict(chk['optimizer'])

    # Setup the aux params.
    epoch = chk['epoch']
    i = chk['i']
    metrics = chk['metrics']
    best_reward = max(metrics['reward'])
    epsilon = chk['epsilon']
    params = Hyperparameters(params=chk['hyperparameters'])
    candidate_name = chk['candidate_name']

    # Refill the sample memory
    print('Filling sample memory')
    learning_model.eval()
    stages = []
    while sample_memory.filled < params.mem_size:
        print("Mem: %d / %d" % (sample_memory.filled, params.mem_size), end='\r')
        done = False
        env, s0, stages = getEnvironment(stages, params)
        while not done:
            action = learning_model.act(torch.tensor(s0, requires_grad=False).to(device), epsilon=epsilon)
            s1, r, done, info = env.step(action)
            s1 = np.array(s1)
            sample_memory.addMemory(s0, action, r, s1, done)
            s0 = s1
    learning_model.train()
    print('Memory fill complete')

    # Return everything we loaded.
    return learning_model, target_model, optimizer, epoch, i, metrics, best_reward, epsilon, params, sample_memory, candidate_name

def trainModel(model_name, checkpoint, params: Hyperparameters):
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
    target_model.eval()
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
    candidate_reward = -np.inf
    candidate_name = 0
    seconds = time.time()
    metrics = {
        'epoch': [0 for _ in range(int(params.epochs/params.test_and_print)+1)],
        'i': [0 for _ in range(int(params.epochs/params.test_and_print)+1)],
        'reward': [0 for _ in range(int(params.epochs/params.test_and_print)+1)],
        'x_movement': [0 for _ in range(int(params.epochs/params.test_and_print)+1)],
        'success': [0 for _ in range(int(params.epochs/params.test_and_print)+1)],
        'iterations': [0 for _ in range(int(params.epochs/params.test_and_print)+1)],
        'time': [0 for _ in range(int(params.epochs/params.test_and_print)+1)],
        'mean': [0 for _ in range(int(params.epochs/params.test_and_print)+1)]
    }

    # Load a checkpoint
    if checkpoint:
        learning_model, target_model, optimizer, epoch, i, metrics, best_reward, epsilon, params, sample_memory, candidate_name = \
            loadCheckpoint(model_name, device, learning_model, target_model, optimizer, sample_memory)

    # Main training loop.
    print('Beginning training')
    while epoch < params.epochs+1:
        # Set the model to training mode and init the full list of stages to pull from.
        learning_model.train()
        
        # Intiialize actions and environment.
        done = False
        env, s0, stages = getEnvironment(stages, params)

        # Track time per 1000 epochs.
        if epoch % 1000 == 0:
            print('Time elapsed (last 1000 epochs): t=%.2fm, e=%d' % ((time.time()-seconds)/60, epoch))
            seconds = time.time()

        # Move through the environment to completion (success or failure).
        for i_ep in count():
            # Get the next action and take it, then record the memory.
            action = learning_model.act(torch.tensor(s0, requires_grad=False).to(device), epsilon=epsilon)
            s1, r, done, info = env.step(action)
            s1 = np.array(s1)
            sample_memory.addMemory(s0, action, r, s1, done)
            s0 = s1

            # Train the network on stored memories.
            if sample_memory.filled > params.batch_size:
                replayMemories(params, learning_model, target_model, sample_memory, loss, optimizer, device)

            # Update the target network and close out the episode, if done.
            i += 1
            if i % params.target_update == 0:
                print("Target update: i=%d" % (i))
                target_model.load_state_dict(learning_model.state_dict())
                target_model.eval()
            if done:
                break

        # Test the current model, checkpoint the model progress, and print status.
        epsilon = max(epsilon*params.explore_decay, params.epsilon_minimum)
        if (epoch % params.test_and_print) == 0:
            testModel(learning_model, params, device, metrics, epoch, i, epsilon)
            i_m = int(epoch/params.test_and_print)

            # Save the model, if it is the current best.
            if metrics['reward'][i_m] > best_reward:
                best_reward = metrics['reward'][i_m]
                print("Model: e=%d, r=%d" % (epoch, best_reward))
                torch.save(learning_model.state_dict(), os.path.join(path, 'model.pth'))
                torch.save(params.dict(), os.path.join(path, 'params.pth'))

            # Save the model, if it is the current best candidate.
            if epoch >= params.candidate_epoch:
                if epoch % 1000 == 0:
                    candidate_reward = -np.inf
                    candidate_name += 1
                if metrics['reward'][i_m] > candidate_reward:
                    candidate_path = os.path.join(path, str(candidate_name))
                    os.makedirs(candidate_path, exist_ok=True)
                    candidate_reward = metrics['reward'][i_m]
                    print("Candidate: c=%d e=%d, r=%d" % (candidate_name, epoch, candidate_reward))
                    torch.save(learning_model.state_dict(), os.path.join(candidate_path, 'model.pth'))
                    torch.save(params.dict(), os.path.join(candidate_path, 'params.pth'))

        doCheckpoint(path, epoch, i, params, metrics, learning_model, target_model, optimizer, loss, epsilon, candidate_name)
        epoch += 1

    # Training done!
    path = os.path.join('save', model_name, 'final')
    os.makedirs(path, exist_ok=True)
    torch.save(learning_model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(params.dict(), os.path.join(path, 'params.pth'))
    saveMetrics(path, metrics, epoch, params)
    print('\n***************************')
    print('Model training complete!')


def demo(model_name, world, stage, demo_scale, checkpoint, record):
    params = Hyperparameters(torch.load(os.path.join('save', model_name, 'params.pth')))
    n_actions = len(actions.action_set[params.actions])
    device = torch.device('cpu')
    model = DQN(params.image_channels, n_actions).to(device)
    if checkpoint:
        chk = torch.load(os.path.join('save', model_name, 'checkpoint.pth'), map_location=device)
        model.load_state_dict(chk['learning_model'])
        print("Model loaded from checkpoint")
    else:
        model.load_state_dict(torch.load(os.path.join('save', model_name, 'model.pth'), map_location=device))
        print("Model loaded from file")
    model.eval()

    video = None
    if record:
        vidFile = os.path.join('save', model_name, 'demo.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(vidFile, fourcc, 60, (256,240))

    viewer = ImageViewer('Super Mario Bot', 240*demo_scale, 256*demo_scale, monitor_keyboard=True, relevant_keys=None, video=video)
    env, render, stages = getEnvironment([(world,stage)], params, viewer=viewer)
    done = False
    action = 0

    # Iterate over the environment until it is over.
    while not done:
        action = model.act(torch.tensor(render, requires_grad=False).to(device), epsilon=0.0)
        state, reward, done, info = env.step(action)
        render = np.array(state)
        if viewer.is_escape_pressed:
            break

    if record:
        video.release()
        cv2.destroyAllWindows()