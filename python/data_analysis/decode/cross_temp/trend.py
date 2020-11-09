import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
ser = pd.read_csv('qdata.csv', index_col=0, squeeze=True)
plt.plot(ser)
plt.show()

X = ser.index
X = np.reshape(X, (len(X), 1))
y = ser.values

model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)
plt.plot(y)
plt.plot(trend)
plt.legend(['data', 'trend'])
plt.show()
