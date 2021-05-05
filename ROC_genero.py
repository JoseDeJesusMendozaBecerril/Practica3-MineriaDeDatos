#### Librerias a utilizar #####
import numpy as np 
import pandas as pd
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #separa data
import sklearn
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score

#KNN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

#logistic regresion
from sklearn.linear_model import LogisticRegression

#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#MatrixConfusion

from sklearn.metrics import confusion_matrix





# Import libraries for graphs
from mpl_toolkits import mplot3d

filename="genero.txt"

def main():
    
    #Lectura de Datos
    data_default = pd.read_csv(filename,sep=",",header=0)

    # Entendimiento de la data
    print('Informacion del data set')
    """ print(data_default.shape)
    print(data_default.head(78))
    print(data_default.columns) """

    #Convertir yes y no a 1 o 0
    data_default.loc[data_default['Gender'] == 'Male' , 'Gender'] = 1
    data_default.loc[data_default['Gender'] == 'Female' , 'Gender'] = 0

    #Imprimir data set despues de conversion
    X_male = data_default['Gender'] == 1
    X_female = data_default['Gender'] == 0
    #### PREPARAR DATA PARA KNN ###

    #Define inputs X columns studen, balance and income
    X = data_default.iloc[:,1:3]

    #Defino output
    y = data_default.iloc[:,0]

    indices = range(X.shape[0])
    #Partimos los data sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_score1 = clf.predict_proba(X_test)[:,1]

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)
    print('roc_auc_score for logistic regresion: ', roc_auc_score(y_test, y_score1))


    knn = KNeighborsClassifier(algorithm='brute',n_neighbors=75)
    knn.fit(X_train,y_train) 

    y_score2 = knn.predict_proba(X_test)[:,1]

    false_positive_rate2, true_positive_rate2, threshold2 = roc_curve(y_test, y_score2)
    print('roc_auc_score for KNN: ', roc_auc_score(y_test, y_score2))


    NB = GaussianNB()
    NB.fit(X_train, y_train)
    y_score3 = NB.predict_proba(X_test)[:,1]

    false_positive_rate3, true_positive_rate3, threshold3 = roc_curve(y_test, y_score3)
    print('roc_auc_score for Naive Bayes: ', roc_auc_score(y_test, y_score3))


    plt.subplots(1, figsize=(10,10))
    plt.title('ROC for data genero')
    
    plt.plot(false_positive_rate1, true_positive_rate1,color="blue")
    plt.plot(false_positive_rate2, true_positive_rate2, color="red")
    plt.plot(false_positive_rate3, true_positive_rate3, color="green")

    plt.plot([0, 1], ls="solid")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(["Logistic regresion", "KNN"," Naive Bayes"])
    plt.show()

main()