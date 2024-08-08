class Scalar:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.label = label
    self._op = _op

  def __repr__(self):
    return f'{self.label}: {self.data}'

  def __add__(self, other):
    out = Scalar(self.data + other.data, (self, other), '+')
    return out
    
  def __mul__(self, other):
    out = Scalar(self.data * other.data, (self, other), '*')
    return out

  def __truediv__(self, other):
    out = Scalar(self.data / other.data, (self, other), '/')
    return out

  def __pow__(self, other):
    out = Scalar(self.data ** other.data, (self, other), 'pow')
    return out
