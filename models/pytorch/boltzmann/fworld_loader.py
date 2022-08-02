import pickle
import random
import torch
from hyperparams import HParams


def get_fworld_data(params: HParams):
    with open('/home/keenan/Downloads/fworld_onpolicysupervised-1uygv488.pkl', 'rb') as f:
        data = pickle.load(f)
        # print(data)

        xs = []
        ys = []
        xlen = 0
        for ii in range(params.num_data):
            datum = data[random.randint(0, len(data) - 1)]
            x = []
            # TODO Potentially exclude some of these layers like V2Wd or Ins if they aren't helpful.
            # With the full list, len(x)==734
            # for layer_name in ["F1", "V2Wd", "V2Fd", "V1F", "S1V", "Ins", "Reward", "S1S", "VL"]:
            for layer_name in ["F1", "Ins", "V2Wd", "V1F", "S1V", "Reward", "S1S", "VL"]:
                x.extend(datum["observations"][layer_name].values)
                # print("Layer length:", layer_name, len(datum["observations"][layer_name].values))
            xlen = len(x)
            xs.append(torch.tensor([x]))
            # 5 different actions in FWorld.
            ys.append(torch.Tensor([[1 if i == datum["chosen_action"] else 0 for i in range(5)]]))
        print("Overall xlen:", xlen)
        return xs, ys, xlen, 5


# test_params = HParams(num_data=5)
# xs, ys, xlen, ylen = get_fworld_data(test_params)
# print(xs)
# print(ys)
# print("end")
