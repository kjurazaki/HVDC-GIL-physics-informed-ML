# Function to count the number of nodes in the expression tree
def count_nodes(expr):
    if expr.is_Atom:
        return 1
    else:
        return 1 + sum(count_nodes(arg) for arg in expr.args)
