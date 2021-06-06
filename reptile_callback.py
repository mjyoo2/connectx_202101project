from stable_baselines3.common.callbacks import BaseCallback
from threading import Thread
from agent import agent

import time
import zmq
import numpy as np
import pickle as pkl
import torch
import copy
import gym

class LowCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, oper_num, port, mode='static', verbose=0):
        super(LowCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.connect('tcp://localhost:{}'.format(port))
        self.request_msg = {'operator_number': oper_num, 'description': 'request'}
        self.recv_sock = ctx.socket(zmq.SUB)
        self.recv_sock.connect('tcp://localhost:{}'.format(port+1))
        self.recv_sock.setsockopt_string(zmq.SUBSCRIBE, '')
        self.updates = 0
        self.mode = mode

    def _on_step(self):
        return True

    def _on_training_start(self) -> None:
        res = pkl.loads(self.recv_sock.recv())
        if res['description'] == 'parameters':
            model_parameter = res['parameters']
            self.model.set_parameters(model_parameter)

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.updates += 1
        if self.updates % 3 == 0:
            model_parameter = self.model.get_parameters()
            msg = {'operator_number': self.request_msg['operator_number'], 'description': 'parameters',
                   'parameters': model_parameter}
            self.sock.send(pkl.dumps(msg))
            res = pkl.loads(self.sock.recv())

            res = pkl.loads(self.recv_sock.recv())
            if res['description'] == 'parameters':
                model_parameter = res['parameters']
                self.model.set_parameters(model_parameter)
        if self.updates % 50 == 0:
            if self.mode == 'learnable':
                opponent = agent(self.model.policy.to('cpu').state_dict())
                self.training_env.env_method('change_opponent', opponent)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        msg = {'operator_number': self.request_msg['operator_number'], 'description': 'finish'}
        self.sock.send(pkl.dumps(msg))

class reptile(object):
    def __init__(self, num_of_operator, port, alpha, env, model):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind('tcp://*:{}'.format(port))
        self.send_sock = ctx.socket(zmq.PUB)
        self.send_sock.bind('tcp://*:{}'.format(port+1))
        self.num_of_operator = num_of_operator
        self.response_msg = {'description': 'response'}
        self.alpha = 0.1
        self.model = model
        self.env = env
        self.test_model = None
        self.update = 0
        time.sleep(1.0)
        model_parameter = self.model.get_parameters()
        msg = {'description': 'parameters', 'parameters': model_parameter}
        self.send_sock.send(pkl.dumps(msg))

    def run(self):
        while True:
            self.update += 1
            num_data = 0
            data = dict()
            while num_data != self.num_of_operator:
                req = pkl.loads(self.sock.recv())
                if req['description'] == 'parameters':
                    data[str(req['operator_number'])] = req
                    num_data += 1
                if req['description'] == 'finish':
                    return
                self.sock.send(pkl.dumps(self.response_msg))

            # merge parameter
            model_parameter = self.model.get_parameters()
            parameter = model_parameter['policy']
            for layer in parameter.keys():
                layer_param = []
                for i in range(self.num_of_operator):
                    layer_param.append(data[str(i)]['parameters']['policy'][layer])
                delta = torch.mean(torch.stack(layer_param), 0)

                parameter[layer] = (1 - self.alpha) * parameter[layer] + self.alpha * delta
            self.model.set_parameters(model_parameter)
            model_parameter = self.model.get_parameters()

            msg = {'description': 'parameters', 'parameters': model_parameter}
            self.send_sock.send(pkl.dumps(msg))

    def save(self, path):
        self.model.save(path)

    def adapt(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
