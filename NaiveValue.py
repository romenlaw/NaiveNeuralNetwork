class Value:
  def __init__(self, data, name=None):
    self.data = data
    self.name = name

  def __repr__(self):
    return f'{self.name}:{self.data}'
    
