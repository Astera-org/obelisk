from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from genpy.optimize.ParameterOptimizer import Client
from genpy.optimize.ttypes import HyperParameter, Suggestions

import random


from parameter_optimizer import HOST
from parameter_optimizer import PORT


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
    optimizer = setup_client()
    hypers = []
    for i in range(5):
        hyper = HyperParameter()
        hyper.name = "val_" + str(i)
        hyper.center = i
        hyper.stddev = 1
        hyper.is_integer = (i % 2 == 0)
        hypers.append(hyper)

    print("Initializing optimizer")
    optimizer.init(hypers, "my hyper search 1", False)
    print("Getting suggestion")
    suggestion = optimizer.suggest()
    print(suggestion)
    optimizer.observe(suggestion.observationId, 7)
    print("All is now fully optimized")
