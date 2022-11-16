import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#import boston.csv
boston = pd.read_csv(r'Assignment2/Boston.csv')
boston = boston.replace('?', np.nan)
boston = boston.dropna()

y = boston['crim']

boston = boston.drop(['Unnamed: 0'], axis=1)
# print(boston)

single_regression_coefficients = pd.DataFrame(columns=['Predictor', 'Coefficient'])

#for each column in the dataframe, we will use linear regression to predict mpg
for col in boston.columns:
    if col != 'crim':
        X = boston[[col]]
        model = LinearRegression().fit(X, y)
        print('Predicting crim using ', col)
        print('Intercept: ', model.intercept_)
        print('Coefficient: ', model.coef_)
        single_regression_coefficients = single_regression_coefficients.append({'Predictor': col, 'Coefficient': model.coef_[0]}, ignore_index=True)
        print('R2 Score: ', r2_score(y, model.predict(X)))
        print()
        #plot the data
        plt.scatter(X, y)
        plt.plot(X, model.predict(X), color='red')
        plt.xlabel(col)
        plt.ylabel('crim')
        #save the plot
        plt.savefig('Assignment2/graphs/' + col + '.png')
        plt.clf()



#fit multiple regression model using all predictors to predict crim
X = boston.drop(['crim'], axis=1)
model = LinearRegression().fit(X, y)
print('Predicting crim using all predictors')
#save the coefficients
multiple_regression_coefficients = pd.DataFrame({'Predictor': X.columns, 'Coefficient': model.coef_})
# find the p values for each predictor
import statsmodels.api as sm
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
print()

#Create a plot displaying the univariate regression coefficients on the x-axis, and the multiple regression coefficients from on the y-axis
plt.scatter(single_regression_coefficients['Coefficient'], multiple_regression_coefficients['Coefficient'])
plt.xlabel('Univariate Regression Coefficient')
plt.ylabel('Multiple Regression Coefficient')
plt.savefig('Assignment2/graphs/UnivariateVsMultipleRegression.png')
plt.clf()


#linear regression with higher order terms
for col in boston.columns:
    if col != 'crim':
        X = pd.DataFrame({'x': boston[col], 'x2': boston[col]**2, 'x3': boston[col]**3})
        model = LinearRegression().fit(X, y)
        print(f'Predicting crim using %s, %s^2, and %s^3' % (col, col, col))
        print('Intercept: ', model.intercept_)
        print('Coefficient: ', model.coef_)
        print('R2 Score: ', r2_score(y, model.predict(X)))
        print()


