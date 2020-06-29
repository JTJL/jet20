import torch

class Config(object):
    __default_config__ = {
        "opt_tolerance" : 1e-3,
        "opt_u" : 10.0,
        "opt_alpha" : 0.1,
        "opt_beta" : 0.5,
        "opt_constraint_tolerance": 1e-5,
        "rouding_precision": 3,
        "device": "cuda",
    }

    def __init__(self,**kwargs):
        self.__dict__.update(**self.__default_config__)
        self.__dict__.update(kwargs)
        self.device = torch.device(self.device)

    def get_namespace(self,namespace):
        rv = {}
        for k, v in self.__dict__.items():
            if not k.startswith(namespace):
                continue
            key = k[len(namespace)+1 :]
            rv[key] = v
        return rv