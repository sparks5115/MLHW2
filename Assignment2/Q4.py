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
print(boston)

#for each column in the dataframe, we will use linear regression to predict mpg
for col in boston.columns:
    if col != 'crim':
        X = boston[[col]]
        model = LinearRegression().fit(X, y)
        print('Predicting crim using ', col)
        print('Intercept: ', model.intercept_)
        print('Coefficient: ', model.coef_)
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
