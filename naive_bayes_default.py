#### Librerias a utilizar #####
import numpy as np 
import pandas as pd
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #separa data
import sklearn
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#MatrixConfusion

from sklearn.metrics import confusion_matrix





# Import libraries for graphs
from mpl_toolkits import mplot3d

filename="Default.txt"




def main():
    
    #Lectura de Datos
    data_default = pd.read_csv(filename,sep="\t",header=0)

    # Entendimiento de la data
    print('Informacion del data set')
    """ print(data_default.shape)
    print(data_default.head(78))
    print(data_default.columns) """

    #Convertir yes y no a 1 o 0
    data_default.loc[data_default['student'] == 'Yes' , 'student'] = 1
    data_default.loc[data_default['student'] == 'No' , 'student'] = 0
    
    data_default.loc[data_default['default'] == 'Yes' , 'default'] = 1
    data_default.loc[data_default['default'] == 'No' , 'default'] = 0

    #### PREPARAR DATA PARA KNN ###

    #Define inputs X columns studen, balance and income
    X = data_default.iloc[:,1:4]

    #Defino output
    y = data_default.iloc[:,0]

    print(X)
    indices = range(X.shape[0])
    #Partimos los data sets
    #X_train,X_test,y_train,y_test = train_test_split(X,y,indices,test_size=0.2)
    X_train,X_test,y_train,y_test,indices_train,indices_test = train_test_split(X, y,indices, test_size=0.2)
    
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_predic = clf.predict(X_test)
    print("y predict", y_predic)
    #get presision
    score = clf.score(X_train,y_train)
    print(score)
    #get matrix confision
    conf_matrix = confusion_matrix(y_test,y_predic)
    print(conf_matrix)

    

main()