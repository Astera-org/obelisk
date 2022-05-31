import sys

import gym
import network
from network.genpy.env.ttypes import ETensor, Action
from network.thrift_agent_server import setup_server


class GymHandler:

    def __init__(self):
        self.name = "AgentZero"

    def init(self, actionSpace, observationSpace):
        self.actionSpace = actionSpace
        self.observationSpace = observationSpace
        return {"name":self.name}

    def step(self, observations, debug):
        print(f'step called obs: {observations} debug: {debug}')

        return {"move":Action(discreteOption=0)}


if __name__ == '__main__':
    handler = GymHandler()
    server = setup_server(handler)
    server.serve()
