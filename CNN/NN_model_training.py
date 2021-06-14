# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import itertools
import plotly.offline as py
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(2)

# from google.colab import files
# Use to load data on Google Colab 
# uploaded = files.upload() # Use to load data on Google Colab 
data = pd.read_csv('/content/drive/MyDrive/data.csv',skipinitialspace=True) 
data.head(100)

col_corr = set()
corr_matrix = data.corr()
for i in range(len(corr_matrix.columns)):
  for j in range(i):
    if(corr_matrix.iloc[i,0]<=0.4)and(corr_matrix.columns[j] not in col_corr):
      colname = corr_matrix.columns[i]
      col_corr.add(colname)
      if colname in data.columns:
        del data[colname]

# Splitting the dataframe into input and outputs
# X is input dataframe wherease Y is output,i.e,diagnosis
X = data.iloc[:,data.columns!="diagnosis"]
Y = data.iloc[:,data.columns =="diagnosis"]

# spliting the dataset into training and testing datasets
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Creating a model with 5 layers
# Activation function sigmoid used since the expected output is binary
model = Sequential([
    Dense(units=16,input_dim=20,activation="relu"),
    Dense(16,activation="relu"),
    Dropout(0.2),
    Dense(16,activation="relu"),
    Dense(16,activation="relu"),
    Dense(1,activation="sigmoid")
])

# summary of the model
model.summary()

# optimizer is 'adam'
# loss function is BinaryCrossentropy()
model.compile(optimizer="adam",loss=tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])

# The model is trained for 25 epochs with batch_size of 15
model.fit(x_train,y_train,batch_size=15,epochs=25)

# Evaluating model for test dataset
pred_nn=model.evaluate(x_test,y_test)

predic = model.predict(x_test)
y_test = pd.DataFrame(y_test)
res_NN = confusion_matrix(y_test,predic.round())

# confusion matrix
plot_confusion_matrix(res_NN,classes=['Benign','Malignant'])
