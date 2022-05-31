from multiprocessing import shared_memory

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from genpy.env.Agent import Client
from genpy.env.ttypes import ETensor, Shape, SpaceSpec


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
    shm_name = res["shm"]
    shm_b = shared_memory.SharedMemory(shm_name)
    print(f'shared mem obj: {shm_b}')

    # write the observation to the shared mem and then call step on the agent

    # TODO: think about whether to encode the entire etensor to bytes and write it or
    # just the values and send an empty etensor in Step just to tell the shape
    values=[1, 2, 3]
    bytes = bytearray(values)
    shm_b.buf[:len(bytes)] = bytes

    # empty values means look in the shared mem
    # but the shape is still passed through
    observation = ETensor(Shape(shape=[1]), values=[])
    observations = {"world": observation}
    
    action = agent.step(observations, "debug_string")
    print(f'action is: {action}')

    # TODO: one side should unlink the shared buffer
    # probably the side that created it? in this case the agent
