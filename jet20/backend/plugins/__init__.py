


class Plugin(object):
    def name(self):
        return self.__class__.__name__

    def preprocess(self,p,x,config):
        raise NotImplementedError("")

    def postprocess(self,p,x,config):
        raise NotImplementedError("")