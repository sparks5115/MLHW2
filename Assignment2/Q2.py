import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

auto = pd.read_csv(r'Assignment2/Auto.csv')
auto = auto.replace('?', np.nan)
auto = auto.dropna()

sns.set_style("whitegrid");
sns.pairplot(auto, height=1);
plt.show()