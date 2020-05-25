from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor

class Scaler:
  def __init__(self, column):
    self.column = column.values.reshape(-1, 1)
    self.scalar = RobustScaler()
    self.scaled_column = None

  @staticmethod
  def get_new_scaler(self):
    return RobustScaler()

  def transform(self):
    self.scaled_column = self.scalar.fit_transform(self.column)
    return self.scaled_column.flatten()

  def inv_transform(self):
    return self.scalar.inverse_transform(self.scaled_column).flatten()
