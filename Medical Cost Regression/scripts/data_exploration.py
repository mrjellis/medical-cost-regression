import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


df = pd.read_csv(r"C:\Users\John's PC\Documents\Python Stuff\Projects\Data Science\Regression\Medical Cost Regression\data\insurance_transformed.csv")

df.corr()['charges'].sort_values(ascending=False)

import seaborn as sns

sns.heatmap(df.corr())

sns.set_style('whitegrid')

g = plt.hist(df['charges'],bins=10)
plt.title('Freq of Charges by Dollars')
plt.ylabel('FREQ #')
plt.xlabel('$ Dollars')

####Questions, 1) what is the average charge by age group
##2) 


#decleares variables
target = 'charges'
y = df[target]
X = df.iloc[:,1:].drop(target,axis=1)

#creates models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)

plt.scatter(X_train['age'],y_train)
#summarizes coeffients
coefs = pd.DataFrame(lm.coef_,X.columns,columns=['Coef']).sort_values('Coef',ascending=False)
predictions = lm.predict(X_test)

#graphs
plt.scatter(y_test,predictions)
sns.distplot(y_test - predictions,bins=50)

#erros measurements
from sklearn.metrics import mean_absolute_error,mean_squared_error
mae = mean_absolute_error(y_test,predictions)
mse = mean_squared_error(y_test,predictions)
rmse = np.sqrt(mean_squared_error(y_test,predictions))


#polynominal transformation on linear regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

quad = PolynomialFeatures(degree=2)
x_quad = quad.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(x_quad, y, test_size=0.33, random_state=0)

plr = LinearRegression().fit(X_train,y_train)

y_train_pred = plr.predict(X_train)
y_test_pred = plr.predict(X_test)

print(plr.score(X_test,y_test))


#Random Forrest Regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

forest = RandomForestRegressor(n_estimators=100,criterion='mse',random_state=1,n_jobs=-1)

forest.fit(X_train,y_train)
forest_train_pred = forest.predict(X_train)

forest_test_pred = forest.predict(X_test)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_train,forest_train_pred)

mean_squared_error(y_test,forest_test_pred)

r2_score(y_train,forest_train_pred)
r2_score(y_test,forest_test_pred)

#plotting

plt.figure(figsize=(12,8))
plt.scatter(forest_train_pred,forest_train_pred-y_train,c='black',marker='o',s=35,alpha=0.5,label='Train Data')
plt.scatter(forest_test_pred,forest_test_pred-y_test,c='red',alpha=0.6)