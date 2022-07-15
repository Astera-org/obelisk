import network.genpy.env.ttypes
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from typing import TypedDict
from network.genpy.env.Agent import Client #see worlds\setup.py
from network.genpy.env.ttypes import ETensor, Shape, SpaceSpec #see worlds\setup.py

import random
import time

# TODO: these should probably be passed in as parameters to setup
# but for now we just use them as defaults
# or we could even define them as constants in thrift
HOST = "127.0.0.1"
PORT = '9090'


def setup_client():
    transport = TSocket.TSocket(HOST, PORT)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    client = Client(protocol)
    # TODO: could make this an enterable thing so can be used like with(client)
    # which takes care of closing it also
    transport.open()
    return client


if __name__ == '__main__':

    agent = setup_client()
    actionSpaceSpec = SpaceSpec(shape=Shape(shape=[1]))
    actionSpace = {"move": actionSpaceSpec}
    observationSpaceSpec = SpaceSpec(shape=Shape(shape=[1]))
    observationSpace = {"world": observationSpaceSpec}
    res = agent.init(actionSpace, observationSpace)
    print(f'init returned: {res}')
    allpairs = []
    import torch
    all_vectors = torch.zeros(size=(30,25))
    for i in range(30):
        vals = [0.0 if random.random() < 0.8 else 1.0 for _ in range(25)]
        observation = ETensor(Shape(shape=[5, 5]), values=vals)
        observations = {"Input": observation, "Output": observation}
        all_vectors[i] = torch.Tensor(vals)
        allpairs.append(observations)
    while True:
        for i in range(30):
            print(allpairs[i])
            action:TypedDict= agent.step(allpairs[i], "debug_string")

            vector_val:torch.Tensor = torch.Tensor(action["Output"].vector.values).float()
            closest_index = torch.argmin((all_vectors - vector_val).abs().sum(dim=1).flatten())

            print(i, closest_index)
            print(f'action is: {action}')
