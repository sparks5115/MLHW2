import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import Carseats.csv
carseats = pd.read_csv(r'Assignment2/Carseats.csv')
carseats = carseats.replace('?', np.nan)
carseats = carseats.dropna()

# Fit a multiple regression model to predict Sales using Price, Urban, and US
X = carseats[['Price', 'Urban', 'US']]
y = carseats['Sales']
X = pd.get_dummies(X, drop_first=True)
regressor = LinearRegression()
regressor.fit(X, y)

# Print the coefficients and intercept of the model
print(f'Coefficients: Price: %s, Urban: %s, US: %s' % (regressor.coef_[0], regressor.coef_[1],  regressor.coef_[2]))
print('Intercept: ', regressor.intercept_)
print('R squared: ', regressor.score(X, y))
#print the 95% confidence interval for each coefficient
print('95% confidence interval for Price: ', np.percentile(regressor.coef_[0], [2.5, 97.5]))
print('95% confidence interval for Urban: ', np.percentile(regressor.coef_[1], [2.5, 97.5]))
print('95% confidence interval for US: ', np.percentile(regressor.coef_[2], [2.5, 97.5]))
print()


# fit a multiple regression model to predict Sales using Price and US
X = carseats[['Price', 'US']]
y = carseats['Sales']
X = pd.get_dummies(X, drop_first=True)
regressor = LinearRegression()
regressor.fit(X, y)

# Print the coefficients and intercept of the model and r squared
print(f'Coefficients: Price: %s, US: %s' % (regressor.coef_[0], regressor.coef_[1]))
print('Intercept: ', regressor.intercept_)
print('R squared: ', regressor.score(X, y))
#print the 95% confidence interval for each coefficient
print('95% confidence interval for Price: ', np.percentile(regressor.coef_[0], [2.5, 97.5]))
print('95% confidence interval for US: ', np.percentile(regressor.coef_[1], [2.5, 97.5]))
print()

# graph the residuals
plt.figure(figsize=(10, 6))
plt.scatter(regressor.predict(X), regressor.predict(X) - y, c='b', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=25)
plt.title('Residual Plot using Price and US')
plt.ylabel('Residuals')
plt.show()





