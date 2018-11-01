# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import math 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

small = 0

#####################################################################
#DATA COLLECTION
#####################################################################
#IMPORT THE DATA
if small == 1: 
    input_dir = "F:\\MNIST\\small"
else:
    input_dir =  "F:\\MNIST\\input"
output_dir = "F:\\MNIST\\output\\"
train_file = "train.csv"
score_file = "test.csv"
scored_file = "scored.csv"
abt= pd.read_csv(os.path.join(input_dir,train_file), encoding='utf8')
score=pd.read_csv(os.path.join(input_dir,score_file), encoding='utf8')


#APPEND ALL THE DATA 
train_objs_num = len(abt)
all_data = pd.concat(objs=[abt, score], axis=0)
all_data_y = pd.DataFrame(all_data['label'])
all_data_x = all_data.drop('label', 1)

# Create a scaler object
sc = StandardScaler()

def poly (mydf):
    num_list = list(mydf.select_dtypes(exclude=['object']).columns)
    for col_name in num_list:
        new_col_name = col_name + '^2'
        mydf[new_col_name] = mydf[col_name] **2
  

abt2 = all_data_x[:train_objs_num]
tmp_y= all_data_y[:train_objs_num]
abt2['label']= tmp_y['label'].values
score2 = all_data_x[train_objs_num:]

train=abt2.sample(frac=0.7,random_state=200)
test=abt2.drop(train.index)

train_y = train.loc[:,['label']]
train_x_uns = train.drop('label', 1)
names = train_x_uns.columns
train_x_np = sc.fit_transform(train_x_uns)
train_x = pd.DataFrame(train_x_np, columns=names)
poly(train_x)

test_y = test.loc[:,['label']]
test_x_uns = test.drop('label', 1)
test_x_np = sc.fit_transform(test_x_uns)
test_x = pd.DataFrame(test_x_np, columns=names)
poly(test_x)


def train_regression(x, y, alpha_param):
    regr = LogisticRegression(penalty="l2", C=alpha_param)
    regr.fit(x, y)
    return regr

def implement_regression(x, y,regr):    
    fit_y = np.asarray(regr.predict(x)) 
    actual_y = np.asarray(y)     
    accuracy = accuracy_score(actual_y,fit_y)
    return accuracy


print ("alpha, train_accuracy, test_accuracy")
#mylist = [.01, .03,1,3,5,7, 10]
mylist = [.01]
for alpha_param in mylist:
    regr=train_regression(train_x, train_y, alpha_param)
    train_accuracy = implement_regression(train_x, train_y, regr)
    test_accuracy = implement_regression(test_x, test_y, regr)
    print (str(alpha_param), str(train_accuracy), str(test_accuracy))

#Hmm have to reattach the column names if want to stick with dataframe. Maybe I dont really need to though
#Handy just to have it as an array for most of the processing? Can push it back into a dataframe later?
