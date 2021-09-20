import seaborn as sbn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


# Linear Regression in nutshell //- ---------------------------------------------------------- -//

# Linear function: y = mx + b

#Gradient Descent for Intercept
def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff  
  return b_gradient

#Gradient Descent for Slope
def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff  
  return m_gradient

# update m, b function with step definding by learning_rate
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]
  
#find best m, b
def gradient_descent(x, y, learning_rate, num_iterations):
  b = 0
  m = 0
  for i in range(num_iterations):
    b, m = step_gradient(b, m, x, y, learning_rate)
  return b, m



# First example of linear function //- ---------------------------------------------------------- -//
'''
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

#get m, b for our real data
b, m = gradient_descent(months, revenue, 0.01, 1000)


y = [m*x + b for x in months]

#build model that describe data
plt.plot(months, revenue, "o")
plt.plot(months, y)

plt.show()
'''
# Second example of linear function //- ---------------------------------------------------------- -//

'''
df = pd.read_csv("new_data.csv")


X = df["height"]
y = df["weight"]


#Build our second model 
plt.plot(X, y, 'o')

b, m = gradient_descent(X, y, num_iterations = 1000, learning_rate = 0.0001)
y_predictions = [ i * m + b for i in X]
plt.plot(X, y_predictions)

plt.show()
'''

# Third example of linear function with sklearn libraty//- ---------------------------------------------------------- -//
"""
#We have imported a dataset of soup sales data vs temperature.
temperature = np.array(range(60, 100, 2))
temperature = temperature.reshape(-1, 1)
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]

plt.plot(temperature, sales, 'o')

#The .fit() method gives the model the line_fitter.coef_, which contains the slope and line_fitter.intercept_, which contains the intercept
line_fitter = LinearRegression().fit(temperature, sales)

#.predict() function to pass in x-values and receive the y-values that this line would predict
sales_predict = line_fitter.predict(temperature)

plt.plot(temperature, sales_predict)
plt.show()
"""

# Fourth example of linear function //- ---------------------------------------------------------- -//

# We’ve loaded in the Boston housing dataset.
# We made the X values the nitrogen oxides concentration (parts per 10 million), and the y values the housing prices.

boston = load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)

# Set the x-values to the nitrogen oxide concentration:
X = df[['NOX']]
# Y-values are the prices:
y = boston.target

#do linear regression on this:

linear = LinearRegression().fit(X, y)
y_predicted = linear.predict(X)

plt.scatter(X, y, alpha=0.4)
# Plot line here:
plt.plot(X,y_predicted, alpha=0.9 )
plt.title("Boston Housing Dataset")
plt.xlabel("Nitric Oxides Concentration")
plt.ylabel("House Price ($)")
plt.show()