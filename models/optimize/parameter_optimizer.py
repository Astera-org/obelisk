from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from genpy.optimize import ParameterOptimizer
from genpy.optimize.ttypes import HyperParameter, Suggestions

from collections import OrderedDict

# NOTE bones must be pulled and installed locally from
# https://gitlab.com/generally-intelligent/bones/
from bones import BONES
from bones import BONESParams
from bones import ObservationInParam
from bones import LinearSpace


HOST = '127.0.0.1'
PORT = '9095'


class BonesHandler:

    def __init__(self):
        self.name = "ParameterOptimizerZero"
        self.id_to_sugg = {}

    def init(self, hyperParameters, runId, restart):
        # TODO Open a matching bones file if there is one
        self.bone_params = BONESParams(
            # TODO These should be arguments
            better_direction_sign=True, initial_search_radius=1,
            is_wandb_logging_enabled=False, resample_frequency=-1
        )
        print("BONES PARAMS:", self.bone_params)

        centers = {}
        space = OrderedDict()
        for param in hyperParameters:
            # TODO Use other HyperParameter values
            centers.update({param.name: param.center})
            space.update([(param.name, LinearSpace(scale=param.stddev))])
        self.bones = BONES(self.bone_params, space)
        self.bones.set_search_center(centers)
        print("CENTERS:", centers)
        print("SPACE:", space)

    def suggest(self):
        sugg = self.bones.suggest().suggestion
        print("GOT SUGGESTION: ", sugg)
        suggestion = Suggestions()
        suggestion.observationId = sugg["suggestion_uuid"]
        suggestion.parameterSuggestions = {}
        for suggName in sugg:
            if isinstance(sugg[suggName], str):
                continue
            suggestion.parameterSuggestions[suggName] = sugg[suggName]
        self.id_to_sugg[suggestion.observationId] = suggestion.parameterSuggestions
        return suggestion

    def observe(self, observationId, value):
        print("GOT OBSERVATION: ", observationId, value)
        output = self.bones.observe(ObservationInParam(input=self.id_to_sugg[observationId], output=value))
        print("OBSERVATION OUTPUT", output)


class ParameterOptimizerHandler:

    def __init__(self):
        self.name = "ParameterOptimizerZero"

    def init(self, hyperParameters, runId, restart):
        pass

    def suggest(self):
        return Suggestions()

    def observe(selfself, observationId, value):
        pass

def setup_server(handler):
    processor = ParameterOptimizer.Processor(handler)
    transport = TSocket.TServerSocket(host=HOST, port=PORT)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    print(f'listening on {HOST}:{PORT}')
    return server


if __name__ == '__main__':
    handler = BonesHandler()
    server = setup_server(handler)
    server.serve()
