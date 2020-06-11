
from jet20.backend.plugins import Plugin

class Simpify(Plugin):
    def __call__(self,p,x,config):
        return p,x
        