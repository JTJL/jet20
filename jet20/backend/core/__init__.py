
from jet20.backend.core.interior_point import interior_point
from jet20.backend.core.primal_dual_interior_point_with_eq import primal_dual_interior_point_with_eq
from jet20.backend.core.primal_dual_interior_point_with_eq_le import primal_dual_interior_point_with_eq_le
from jet20.backend.core.primal_dual_interior_point_with_le import primal_dual_interior_point_with_le
from jet20.backend.core.status import *




def solve(p,x,config,fast,should_stops=None,duals=None):
    if not p.obj:
        raise Exception("no obj found")

    if p.eq is not None and p.le is not None:
        return primal_dual_interior_point_with_eq_le(x,p.obj,p.le,p.eq,should_stops,fast=fast,duals=duals,**config.get_namespace("opt"))

    elif p.eq is not None:
        return primal_dual_interior_point_with_eq(x,p.obj,p.eq,should_stops,fast=fast,duals=duals,**config.get_namespace("opt"))

    elif p.le is not None:
        return primal_dual_interior_point_with_le(x,p.obj,p.le,should_stops,fast=fast,duals=duals,**config.get_namespace("opt"))
    
    else:
        raise Exception("no constraits found")

