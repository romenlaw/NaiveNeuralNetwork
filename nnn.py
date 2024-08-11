import math

class Scalar:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.label = label
    self._op = _op
    self._operands = _children
    self.grad = 0.
    # will be set by operator magic functions
    self._backward = lambda: None

  def __repr__(self):
    return f'{self.label}: {self.data}'

  def backward(self):
    nodes=[] 

    # need to traverse to the leaf nodes before adding to the list
    def build(n):
      if n not in nodes:
        for child in n._operands:
          build(child)
        nodes.append(n) 
      
    build(self)
    # reverse order of list, i.e. root to leaves
    nodes.reverse()
    print(nodes)
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

class Neuron:
  def __init__(self, nin):
    """
    nin - number of inputs to the neuron
    """
    self.w = [Scalar(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Scalar(random.uniform(-1, 1))
    return

  def __call__(self, X):
    """
    forward pass of the neuron, X and W must be same dimension
    returns tanh( W*X+b )
    """
    out = sum(([xi*wi for xi, wi in list(zip(X, self.w))]), self.b)
    return out.tanh()

class Layer:
  def __init__(self, nin, nout):
    """
    nin - number of inputs of the layer - i.e. number of neurons in input layer
    nout - number of outputs of the layer - i.e. number of neurons of current layer
    """
    self.neurons = [Neuron(nin) for _ in (range(nout))]
    return

  def __call__(self, X):
    """
    forward pass of all the neurons in the current layer
    """
    outs = [n(X) for n in self.neurons]
    return outs

class MLP:
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
    self.layers = [Layer(nin, nout) for nin, nout in list(zip(nins, nouts))]
    return

  def __call__(self, X):
    """
    forward pass of the MLP, X dimension is nin
    """
    for layer in self.layers:
      X = layer(X)

    return X
