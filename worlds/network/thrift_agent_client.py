from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from network.genpy.env.Agent import Client
from network.genpy.env.ttypes import ETensor, Shape, SpaceSpec

import random


# TODO: these should probably be passed in as parameters to setup
# but for now we just use them as defaults
# or we could even define them as constants in thrift
HOST = '127.0.0.1'
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
    observation = ETensor(Shape(shape=[5, 5]), values=[random.random() for _ in range(25)])
    observations = {"Input": observation, "Output": observation}
    action = agent.step(observations, "debug_string")
    print(f'action is: {action}')
