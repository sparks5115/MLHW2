import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv(r'Assignment2/Auto.csv')
df = df.replace('?', np.nan)
df = df.dropna()

hp_mpg = df[['horsepower', 'mpg']]

x = hp_mpg[['horsepower']]
y = hp_mpg.mpg

model = LinearRegression().fit(x, y)

print('Intercept: ', model.intercept_)
print('Coefficient: ', model.coef_)
print('R2 Score: ', r2_score(y, model.predict(x)))

print('Predicted mpg for 95hp: ', model.predict([[95]]))


list = []
for l in x.values:
    list.append(int(l[0]))

hp_mpg_plt = plt.axes()

hp_mpg_plt.scatter(list, y)
hp_mpg_plt.set_xlabel('Horsepower')
hp_mpg_plt.set_ylabel('Miles per Gallon')
hp_mpg_plt.set_title('Horsepower vs Miles per Gallon')

hp_mpg_plt.plot(list, model.predict(x), color='red')
plt.show()
