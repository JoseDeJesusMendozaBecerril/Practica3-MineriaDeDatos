#### Librerias a utilizar #####
import numpy as np 
import pandas as pd
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #separa data

from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

#KNN
from sklearn.neighbors import NearestNeighbors




# Import libraries for graphs
from mpl_toolkits import mplot3d

filename="Default.txt"




def main():
    
    #Lectura de Datos
    data_default = pd.read_csv(filename,sep="\t",header=0)

    # Entendimiento de la data
    print('Informacion del data set')
    print(data_default.shape)
    print(data_default.head(78))
    print(data_default.columns)

    #Convertir yes y no a 1 o 0
    data_default.loc[data_default['student'] == 'Yes' , 'student'] = 1
    data_default.loc[data_default['student'] == 'No' , 'student'] = 0
    
    data_default.loc[data_default['default'] == 'Yes' , 'default'] = 1
    data_default.loc[data_default['default'] == 'No' , 'default'] = 0

    #Imprimir data set despues de conversion
    #print(data_default)

    print("######### KNN #########")



    #### PREPARAR DATA PARA KNN ###

    #Defino entradas X Solamente la columna 6
    X = data_default.iloc[:,2:4]
    print(X)


    #Defino Y
    y = data_default.iloc[:,0]
    print(y)


    #Partimos los data sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    
    #Defino el algoritmo a usar
    neigh = NearestNeighbors(algorithm='auto',n_neighbors=2)


    #Entreno al modelo
    n = neigh.fit(X_train,y_train)
    
    #distances,indices = n.kneighbors(X_train)
    #print(distances)
    #y_test = y_test.values.reshape(1,-1)
    #print(y_test)

    #result = neigh.kneighbors(y_train[0].values,2,return_distance=False)
    #print(result)
    #precision = 0
    
    




main()