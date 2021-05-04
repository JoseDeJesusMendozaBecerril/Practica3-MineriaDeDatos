#### Librerias a utilizar #####
import numpy as np 
import pandas as pd
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #separa data
import sklearn
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

#KNN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

#MatrixConfusion

from sklearn.metrics import confusion_matrix





# Import libraries for graphs
from mpl_toolkits import mplot3d

filename="Default.txt"
filename2="genero.txt"




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

    #Imprimir data set despues de conversion
    #print(data_default)

    print("######### KNN to Default #########")

    #### PREPARAR DATA PARA KNN ###

    #Define inputs X columns studen, balance and income
    X = data_default.iloc[:,1:4]

    #Defino output
    y = data_default.iloc[:,0]


    #Partimos los data sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    values = [1,2,3,5,10,15,20,50,75,100]
    data_Presition=[]
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    
    


    for a in values:
        
        knn = KNeighborsClassifier(algorithm='brute',n_neighbors=a)
        
        knn.fit(X_train,y_train) 
        y_predic = knn.predict(X_test)
        print(a)
        data_Presition.append(knn.score(X_test,y_test))
        conf_matrix = confusion_matrix(y_test,y_predic)
        print(conf_matrix)


    x = values
    y = data_Presition

    plt.scatter(x, y)
    plt.title("KNN Clasifier")
    plt.xlabel("Neighbors")
    plt.ylabel("Precision")
    plt.legend(loc='upper left')
    plt.show()
   
    print("######### KNN to genero #########")

    
main()