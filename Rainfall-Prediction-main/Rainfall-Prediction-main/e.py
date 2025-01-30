
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Rainfall.csv')
df.head()

df.shape

df.info()

df.describe().T

df.isnull().sum()

df.columns

df.rename(str.strip,
          axis='columns',
          inplace=True)

df.columns

for col in df.columns:

  # Checking if the column contains
  # any null values
  if df[col].isnull().sum() > 0:
    val = df[col].mean()
    df[col] = df[col].fillna(val)

df.isnull().sum().sum()

plt.pie(df['rainfall'].value_counts().values,
        labels = df['rainfall'].value_counts().index,
        autopct='%1.1f%%')
plt.show()

df.groupby('rainfall').mean()

features = list(df.select_dtypes(include = np.number).columns)
features.remove('day')
print(features)

plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.distplot(df[col])
plt.tight_layout()
plt.show()

plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.boxplot(df[col])
plt.tight_layout()
plt.show()

df.replace({'yes':1, 'no':0}, inplace=True)

plt.figure(figsize=(10,10))
sb.heatmap(df.corr() > 0.8,
           annot=True,
           cbar=False)
plt.show()

df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)

features = df.drop(['day', 'rainfall'], axis=1)
target = df.rainfall

X_train, X_val, Y_train, Y_val = train_test_split(features,
                                      target,
                                      test_size=0.2,
                                      stratify=target,
                                      random_state=2)

# As the data was highly imbalanced we will
# balance it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority',
                        random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)

# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

for i in range(3):
  models[i].fit(X, Y)

  print(f'{models[i]} : ')

  train_preds = models[i].predict_proba(X)
  print('Training Accuracy : ', metrics.roc_auc_score(Y, train_preds[:,1]))

  val_preds = models[i].predict_proba(X_val)
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, val_preds[:,1]))
  print()

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

# Use ConfusionMatrixDisplay.from_estimator instead of plot_confusion_matrix
ConfusionMatrixDisplay.from_estimator(models[2], X_val, Y_val)
plt.show()

print(metrics.classification_report(Y_val,
                                    models[2].predict(X_val)))

import pickle

# Choose the model you want to save, for example, XGBoost Classifier
model_to_save = models[1]  # This refers to the XGBClassifier

# Save the model
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model_to_save, file)

import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
import os

model_path = 'xgboost_model.pkl'  # Adjust to your saved model filename

# Load the trained model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = pickle.load(open(model_path, 'rb'))
model_to_save = models[1]  # This refers to the XGBClassifier

# Save the model
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model_to_save, file)