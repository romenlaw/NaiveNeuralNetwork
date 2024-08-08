import graphviz

def trace(root):
  nodes=[] 
  edges=set()

  def build(n):
    if n not in nodes:
      nodes.insert(0, n) # this is useless because graphvis does not preserve order when laying out complex graph
      for child in tuple(reversed(n._operands)):
        edges.add( (child, n) ) 
        build(child)

  build(root)
  return nodes, edges


def visualise(root, format='svg', rankdir='LR'):
  """
  Visulaise a computabional graph.
  
  format: png | svg | ...
  rankdir: TB (top to bottom) | LR (left to right)
  """
  assert rankdir in ['TB', 'LR']
  nodes, edges = trace(root)
  g = graphviz.Digraph(format='svg', graph_attr={'rankdir': 'LR'})
  for n in nodes:
    node_name=str(id(n))
    g.node(name=node_name, label="{%s.data=%.4f\\ngrad=%.4f}" % (n.label, n.data, n.grad), shape="record")
    if n._op:
      op_node_name=str(id(n))+n._op
      g.node(name=op_node_name, label=n._op)
      g.edge(op_node_name, node_name)

  for n1, n2 in edges:
    g.edge(str(id(n1)), str(id(n2))+n2._op)

  return g