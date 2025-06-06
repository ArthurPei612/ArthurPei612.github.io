def dot_product(X, Y):
  """Calculates the dot product of two 2D vectors.

  Args:
    X: A list representing the first vector, e.g., [x1, x2].
    Y: A list representing the second vector, e.g., [y1, y2].

  Returns:
    The dot product (scalar value).
  """
  if len(X) != 2 or len(Y) != 2:
    raise ValueError("Vectors must be 2-dimensional.")
  return X[0] * Y[0] + X[1] * Y[1]

def cross_product_2d(X, Y):
  """Calculates the 2D cross product (z-component) of two vectors.

  Args:
    X: A list representing the first vector, e.g., [x1, x2].
    Y: A list representing the second vector, e.g., [y1, y2].

  Returns:
    The z-component of the cross product (scalar value).
  """
  if len(X) != 2 or len(Y) != 2:
    raise ValueError("Vectors must be 2-dimensional.")
  return X[0] * Y[1] - X[1] * Y[0]
