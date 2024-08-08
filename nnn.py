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
    # reuse the code from visualiser
    nodes=[] 

    def build(n):
      if n not in nodes:
        nodes.append(n) 
        for child in tuple(reversed(n._operands)):
          build(child)

    build(self)
    print(nodes)
    self.grad=1
    for n in nodes:
      n._backward()

  def __add__(self, other):
    print(self)
    print(other, isinstance(other, Scalar), type(other))
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
    other = other if isinstance(other, Scalar) else Scalar(other)
    out = Scalar(self.data / other.data, (self, other), '/')
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)) , "only int or float allowed"
    out = Scalar(self.data ** other, (self,), 'pow')

    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out

