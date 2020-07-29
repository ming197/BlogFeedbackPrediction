import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math

filename = "blogData_train.csv"
train_data = pd.read_csv(filename,header=None)
#train_data = train_data.iloc[np.random.permutation(len(train_data))]
train_output = train_data[len(train_data.columns)-1]
del train_data[len(train_data.columns)-1]

filename = "blogData_test-2012.02.01.00_00.csv"
test_data = pd.read_csv(filename,header=None)
test_output = test_data[len(test_data.columns)-1]
del test_data[len(test_data.columns)-1]
rank_test = [index for index,value in sorted(list(enumerate(test_output)),key=lambda x:x[1], reverse=True)]

reg = LinearRegression()
rf = RandomForestRegressor()
gradBoost = GradientBoostingRegressor()
ada = AdaBoostRegressor()



regressors = [reg,rf,gradBoost,ada]
regressor_names = ["Linear Regression","Random Forests","Gradient Boosting","Adaboost"]



for regressor,regressor_name in zip(regressors,regressor_names):
    
    regressor.fit(train_data,train_output)
    predicted_values = regressor.predict(test_data)
    rank_predict = [index for index,value in sorted(list(enumerate(predicted_values)),key=lambda x:x[1], reverse=True)]
    counter = len([x for x in rank_predict[:10] if x in rank_test[:10]])

    print ("Mean Squared Error for ",regressor_name, " : ", metrics.mean_squared_error(test_output,predicted_values))
    print ("R2 score for ",regressor_name, " : ",metrics.r2_score(test_output,predicted_values))
    print ("HIT@10 for ",regressor_name, " : ",counter)
    print("\n")

