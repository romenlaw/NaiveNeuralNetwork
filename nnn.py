import math

class Scalar:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.label = label
    self._op = _op
    self._operands = set(_children) # using set for efficiency?
    self.grad = 0.
    # will be set by operator magic functions
    self._backward = lambda: None

  def __repr__(self):
    return f'{self.label}: {self.data}'

  def backward(self):
    nodes=[] 
    visited=set()

    # need to traverse to the leaf nodes before adding to the list
    def build(n):
      if n not in nodes:
        if n not in visited:
          visited.add(n)
          for child in n._operands:
            build(child)
          nodes.append(n) 
      
    build(self)
    # reverse order of list, i.e. root to leaves
    nodes.reverse()
    #print(nodes)
    self.grad=1
    for n in nodes:
      n._backward()

  def relu(self):
    out = Scalar(self.data if self.data>=0 else 0, _children=(self,) , _op='ReLU')

    def _backward():
      # += either out.grad or 0
      self.grad += (out.data>0) * out.grad

    out._backward = _backward

    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
    out = Scalar(t, (self,), 'tanh')

    def _backward():
      self.grad += (1-t**2) * out.grad
    out._backward = _backward
    return out

  def exp(self):
    x = self.data
    out = Scalar(math.exp(x), (self,), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  def __add__(self, other):
    #print(self)
    #print(other, isinstance(other, Scalar), type(other))
    other = other if isinstance(other, Scalar) else Scalar(other)
    out = Scalar(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward
    return out
    
  def __mul__(self, other):
    other = other if isinstance(other, Scalar) else Scalar(other)
    out = Scalar(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out

  def __truediv__(self, other):
    return self * other**(-1)

  def __pow__(self, other):
    assert isinstance(other, (int, float)) , "only int or float allowed"
    out = Scalar(self.data ** other, (self,), f'pow({other})')

    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

  def __neg__(self): # -self
    return self * -1

  def __radd__(self, other): # other + self
    return self + other

  def __sub__(self, other): # self - other
    return self + (-other)

  def __rsub__(self, other): # other - self
    return other + (-self)

  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __rtruediv__(self, other): # other / self
    return other * self**-1

import random

class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad=0.

  def parameters(self):
    return []

class Neuron(Module):
  def __init__(self, nin, nonlin=True, _layer='', _index=''):
    """
    nin - number of inputs to the neuron
    layer - name of the layer where the neuron sits. used for visualisation.
    """
    self.w = [Scalar(random.uniform(-1, 1), label=f"w(L{_layer},{_index},I{i})") for i in range(nin)]
    self.b = Scalar(random.uniform(-1, 1), label=f'b(L{_layer},{_index})')
    self.non_linear=nonlin
    self._layer=_layer
    self._index=_index
    return

  def __call__(self, X):
    """
    forward pass of the neuron, X and W must be same dimension
    returns tanh( W*X+b )
    """
    out = sum(([xi*wi for xi, wi in list(zip(X, self.w))]), self.b)
    out.label=f'o(L{self._layer},{self._index})'
    return out.tanh() if self.non_linear else out

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self) -> str:
    return f"Neuron[L{self._layer},{self._index},non_linear={self.non_linear}]"

class Layer(Module):
  def __init__(self, nin, nout, nonlin=True, label=''):
    """
    nin - number of inputs of the layer - i.e. number of neurons in input layer
    nout - number of outputs of the layer - i.e. number of neurons of current layer
    """
    self.neurons = [Neuron(nin, nonlin, label, str(i)) for i in (range(nout))]
    self.label=label
    self.non_linear = nonlin
    self.nin=nin
    self.nout=nout
    return

  def __call__(self, X):
    """
    forward pass of all the neurons in the current layer
    """
    outs = [n(X) for n in self.neurons]
    return outs[0] if len(outs)==1 else outs

  def __repr__(self) -> str:
    return f"\n Layer[{self.label},{self.nin} x {self.nout}, non_linear={self.non_linear}]: " +\
      f"[{', '.join(str(n) for n in self.neurons)}]"

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
    # equivalent to:
    #params=[]
    #for n in self.neurons:
    #  params.extend( n.parameters() )
    #return params


class MLP(Module):
  """
  multi-layer perceptron
  """
  def __init__(self, nin, nouts):
    """
    nin - number of input neurons
    nouts - list of nout. The len(nouts) is the number of layers
            each element of nouts is number of neurons in corresponding layer
    """
    nins = [nin] + nouts[:-1]
    # only the last layer is non-linear
    self.layers = [Layer(nin, nout, index<len(nouts)-1, str(index)) \
      for index, (nin, nout) in enumerate(list(zip(nins, nouts)))]

    self.nin=nin
    return

  def __call__(self, X):
    """
    forward pass of the MLP, X dimension is nin
    """
    # turn X into list of Scalar to make visulaisation clearer
    if not isinstance(X[0], Scalar):
      X=[Scalar(X[i], label=f"x{i}") for i in range(len(X))]
    
    for layer in self.layers:
      X = layer(X)

    return X

  def parameters(self):
    return [p for l in self.layers for p in l.parameters()]

  def __repr__(self) -> str:
    return f"MLP[nin={self.nin}, n_layers={len(self.layers)}]: " + \
      f"[{', '.join(str(layer) for layer in self.layers)}]"
