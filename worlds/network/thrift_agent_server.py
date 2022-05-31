from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from network.genpy.env import Agent
from network.genpy.env.ttypes import Action, ETensor, Shape


HOST = '127.0.0.1'
PORT = '9090'


class AgentHandler:

    def __init__(self):
        self.name = "AgentZero"

    def init(self, actionSpace, observationSpace):
        print(f'init called actionsSpace: {actionSpace} observationSpace: {observationSpace}')
        self.moveActionSpace = actionSpace["move"]
        self.discreteLabels = self.moveActionSpace.discreteLabels
        print(f'discrete labels {self.discreteLabels}')
        self.observationSpace = observationSpace
        return {"name": self.name}

    def step(self, observations, debug):
        print(f'step called obs: {observations} debug: {debug}')
        if self.discreteLabels:
            action = Action(discreteOption=0)
        else:
            action = Action(vector=ETensor(shape=Shape(shape=[1]), values=[1.0]))
        return {"move": action}

def setup_server(handler):
    processor = Agent.Processor(handler)
    transport = TSocket.TServerSocket(host=HOST, port=PORT)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    print(f'listening on {HOST}:{PORT}')
    return server


if __name__ == '__main__':
    handler = AgentHandler()
    server = setup_server(handler)
    server.serve()
