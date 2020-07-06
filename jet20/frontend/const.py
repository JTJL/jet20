OP = str

OP_EQUAL: OP = "=="
OP_LT: OP = "<"
OP_LE: OP = "<="
OP_GT: OP = ">"
OP_GE: OP = ">="

OP_PAIRS = {OP_GT: OP_LT, OP_GE: OP_LE}  # {">":"<", ">=":"<="}
