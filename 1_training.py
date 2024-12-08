import pandas as pd
data=pd.read_csv("Crop_recommendation.csv")

print("looking for NAN values:",data.columns[data.isna().any()])
print("looking for duplicate values:",data.duplicated().any())

print(f"shape of data is:",data.shape)
print(f"size of the data is:",data.size)
print(f"info of data is:",data.info())
print(data.describe())

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
print('[info]data segregation complete...')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=100,
    test_size=0.2
)
print('[info]data splite into train and test partitions...')
print("unique values are:",data['label'].unique())

data['label']=data['label'].map(
    {
        'rice':1,
        'maize':2,
        'chickpea':3,
        'kidneybeans':4,
        'pigeonpeas':5,
        'mothbeans':6,
        'mungbean':7,
        'blackgram':8,
        'lentil':9,
        'pomegranate':10,
        'banana':11,
        'mango':12,
        'grapes':13,
        'watermelon':14,
        'muskmelon':15,
        'apple':16,
        'orange':17,
        'papaya':18,
        'coconut':19,
        'cotton':20,
        'jute':21,
        'coffee':22
    }
)
print(data.columns[data.isna().any()])
print("unique values after encoding are:",data['label'].unique())

from sklearn.svm import SVC#importing algo
svm=SVC()#initilizing algo
svm.fit(x_train,y_train)#training algo on train partition
print('[info]model training complete...')

svm_pred=svm.predict(x_test)

from sklearn.metrics import classification_report
model_parameters=classification_report(y_test,svm_pred)
print('MODEL EVALUATION METRIC:\n',model_parameters)
from sklearn.metrics import confusion_matrix
svm_cf=confusion_matrix(y_test,svm_pred)
print("confusion matrix :\n",svm_cf)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(svm_cf,fmt='d',cmap='Blues',annot=True)
plt.xlabel('PRIDICTED')
plt.ylabel('ACTUAL')
plt.show()

N=float(input("enter N:"))
P=float(input('enter p:'))
K=float(input("enter k:"))
temperature=float(input("enter temperature:"))
humidity=float(input("enter humidity:"))
ph=float(input("enter ph:"))
rainfall=float(input("enter rainfall:"))

user_inputs=[[N,P,
              K,temperature,humidity,ph,rainfall]]

output=svm.predict(user_inputs)[0]
print(f"for given user inputs the predicted species is:{output}")

import joblib
joblib.dump(svm,'trained2_svm.pkl')
print('[info] tarined model saving to hard disk...')