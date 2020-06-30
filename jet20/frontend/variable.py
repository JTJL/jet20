from jet20.frontend.expression import Expression


class Variable(Expression):
    """

    """
    def __init__(self, index: int, name: str):
        Expression.__init__(self, None, var_index=index)
        self._index: int = index
        self._name: str = name
        # TODO: upper/lower bound, constraint_type([0,1], integer, float...)

    @property
    def index(self) -> int:
        return self._index

    @property
    def name(self) -> str:
        return self._name

    def __hash__(self):
        pass

    def __str__(self):
        return f"{super().__str__()} (expr), {self.name} (name), {self.index} (index)"
