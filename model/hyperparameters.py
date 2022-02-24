class Hyperparameters(object):

    _all_stages = [(w, l) for w in range(1, 9) for l in range(1, 5)]
    _dry_stages = [
        (1,1), (1,2), (1,3), (1,4), 
        (2,1), (2,3), (2,4), 
        (3,1), (3,2), (3,3), (3,4), 
        (4,1), (4,2), (4,3), (4,4), 
        (5,1), (5,2), (5,3), (5,4), 
        (6,1), (6,2), (6,3), (6,4), 
        (7,1), (7,3), (7,4),
        (8,1), (8,2), (8,3), (8,4), 
    ]

    def __init__(self, params={}) -> None:
        self._epochs = self._getParam(params, 'epochs', 200000)
        self._frame_skip = self._getParam(params, 'frame_skip', 4)
        self._image_channels = self._getParam(params, 'image_channels', 4)
        self._mem_size = self._getParam(params, 'mem_size', 250000)
        self._actions = self._getParam(params, 'actions', 'SIMPLE_MOVEMENT') # see environment.actions
        self._explore_decay = self._getParam(params, 'explore_decay', 0.99997)
        self._epsilon_minimum = self._getParam(params, 'epsilon_minimum', 0.1)
        self._gamma = self._getParam(params, 'gamma', 0.95)
        self._learning_rate = self._getParam(params, 'learning_rate', 0.00025)
        self._batch_size = self._getParam(params, 'batch_size', 32)
        self._target_update = self._getParam(params, 'target_update', 10000) # iterations
        self._metrics_save = self._getParam(params, 'metrics_save', 50) # epochs
        self._checkpoint = self._getParam(params, 'checkpoint', 250) # epochs
        self._test_and_print = self._getParam(params, 'test_and_print', 1) # epochs
        self._clip_reward = self._getParam(params, 'clip_reward', False)
        self._stages = self._dry_stages
        self._test_stage = self._getParam(params, 'test_stage', (8,1))

    def dict(self):
        return {
            'epochs': self._epochs,
            'frame_skip': self._frame_skip,
            'image_channels': self._image_channels,
            'mem_size': self._mem_size,
            'actions': self._actions,
            'explore_decay': self._explore_decay,
            'epsilon_minimum': self._epsilon_minimum,
            'gamma': self._gamma,
            'learning_rate': self._learning_rate,
            'batch_size': self._batch_size,
            'target_update': self._target_update,
            'metrics_save': self._metrics_save,
            'checkpoint': self._checkpoint,
            'test_and_print': self._test_and_print,
            'clip_reward': self._clip_reward,
            'stages': self._stages,
            'test_stage': self._test_stage
        }
    
    @property
    def epochs(self):
        return self._epochs
    
    @property
    def frame_skip(self):
        return self._frame_skip
    
    @property
    def image_channels(self):
        return self._image_channels
    
    @property
    def mem_size(self):
        return self._mem_size
    
    @property
    def actions(self):
        return self._actions
    
    @property
    def explore_decay(self):
        return self._explore_decay
    
    @property
    def epsilon_minimum(self):
        return self._epsilon_minimum
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def learning_rate(self):
        return self._learning_rate
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def target_update(self):
        return self._target_update
    
    @property
    def metrics_save(self):
        return self._metrics_save
    
    @property
    def checkpoint(self):
        return self._checkpoint
    
    @property
    def test_and_print(self):
        return self._test_and_print
    
    @property
    def clip_reward(self):
        return self._clip_reward
    
    @property
    def stages(self):
        return self._stages.copy()
    
    @property
    def test_stage(self):
        return self._test_stage

    def _getParam(self, params, name, default):
        return params[name] if name in params else default