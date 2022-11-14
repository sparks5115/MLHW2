import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

auto = pd.read_csv(r'Assignment2/Auto.csv')
auto = auto.replace('?', np.nan)
auto = auto.dropna()

# comenting this out because it takes a long time to run
# sns.set_style("whitegrid");
# sns.pairplot(auto, height=1);
# plt.show()

correlation = auto.corr(numeric_only=True)
# print(correlation)
# correlation.to_csv(r'Assignment2/correlation.csv')

# this is our response variable, it will stay the same
y = auto.mpg

#create 3 subplots

# for each column in the dataframe, we will use linear regression to predict mpg
for column in auto:
    if column != 'mpg' and column != 'name':
        print('Predicting mpg using ', column)
        x = auto[[column]]
        model = LinearRegression().fit(x, y)
        print('Intercept: ', model.intercept_)
        print('Coefficient: ', model.coef_)
        print('R2 Score: ', r2_score(y, model.predict(x)))
        print()

        # plot the data
        fig, axs = plt.subplots(3)
        x_for_plot = []
        for l in x.to_numpy():
            x_for_plot.append(float(l[0]))
        y_for_plot = y

        #plot scatterplot and regression line on axs[0]
        axs[0].scatter(x_for_plot, y_for_plot)
        axs[0].set_xlabel(str(column))
        axs[0].set_ylabel('Miles per Gallon')
        axs[0].plot(x_for_plot, model.predict(x), color='red')

        #create a residual plot on axs[1]
        residuals = y - model.predict(x)
        axs[1].scatter(x_for_plot, residuals)
        axs[1].set_xlabel(str(column))
        axs[1].set_ylabel('Residuals')
        
        #create leverage plot on axs[2]
        leverage = model.predict(x) - model.predict(x).mean()
        axs[2].scatter(x_for_plot, leverage)
        axs[2].set_xlabel(str(column))
        axs[2].set_ylabel('Leverage')

        #save the plot
        plt.savefig(r'Assignment2/' + str(column) + '.png')
        plt.clf()
