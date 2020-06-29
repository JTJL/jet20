from functools import wraps
from time import time

import logging
logger = logging.getLogger(__name__)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.debug('func:%s took: %2.4f sec',f.__name__,te-ts)
        return result
    return wrap


class NotImproving(object):
    def __init__(self,episilon=0.0,max_count=1):
        super(NotImproving,self).__init__()
        self.episilon = episilon
        self.max_count = max_count
        self.last_norm = None
        self.count = 0

    def __call__(self,norm):
        if self.last_norm is None:
            self.last_norm = norm
            return False
        
        if self.last_norm - norm <= self.episilon:
            self.count += 1
        else:
            self.count = 0
        
        if self.count >= self.max_count:
            return True
        
        self.last_norm = norm
        return False