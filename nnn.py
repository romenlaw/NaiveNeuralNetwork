class Scalar:
  def __init__(self, data, _op='', label=''):
    self.data = data
    self.label = label
    self._op = _op

  def __repr__(self):
    return f'{self.label}: {self.data}'

  def __add__(self, other):
    out = Value(self.data + other.data, '+')
    return out
    
  def __mul__(self, other):
    out = Value(self.data * other.data, '*')
    return out

  def __truediv__(self, other):
    out = Value(self.data / other.data, '/')
    return out

  def __pow__(self, other):
    out = Value(self.data ** other.data, 'pow')
    return out
