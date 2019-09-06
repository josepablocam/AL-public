# visualize traces collected
from graphviz import Digraph


def to_dot(trace, title=None):
    dot = Digraph(comment=title)
    for elem in trace:
        dot.node(
            str(elem['call_id']),
            '%s.%s' % (elem['func_module'], elem['func_name'])
        )
        for dep in elem['dependencies']:
            if dep != -1:
                dot.edge(str(dep), str(elem['call_id']))
    return dot
