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


#logistic regresion
from sklearn.linear_model import LogisticRegression


#Naive Bayes
from sklearn.naive_bayes import GaussianNB


# Import libraries for graphs
from mpl_toolkits import mplot3d

filename="Default.txt"
filename2="genero.txt"




def main():
    
                # ---------------------CLASIFICACION CON KNN ------------------
                                        #DATA SET default.txt
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
        #DEFINICION DEL MODELO   
        knn = KNeighborsClassifier(algorithm='brute',n_neighbors=a)

        #ENTRENAMIENTO
        knn.fit(X_train,y_train) 
        y_predic = knn.predict(X_test)
        print(a)
        
        #CALCULO PRECISIONES
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
   

                                ## Data set Genero.txt

    #Lectura de Datos
    data_default = pd.read_csv(filename2,sep=",",header=0)

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
    #X_train,X_test,y_train,y_test = train_test_split(X,y,indices,test_size=0.2)
    X_train,X_test,y_train,y_test,indices_train,indices_test = train_test_split(X, y,indices, test_size=0.2)
    
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    values = [1,2,3,5,10,15,20,50,75,100]
    data_Presition=[]

    for a in values:
        
        knn = KNeighborsClassifier(algorithm='brute',n_neighbors=a)
        
        knn.fit(X_train,y_train) 
        y_predic = knn.predict(X_test)
        print(a)
        data_Presition.append(knn.score(X_test,y_test))
        conf_matrix = confusion_matrix(y_test,y_predic)
        print(conf_matrix)
    
    """ x = values
    y = data_Presition

    plt.scatter(x, y)
    plt.title("KNN Clasifier")
    plt.xlabel("Neighbors")
    plt.ylabel("Precision")
    plt.legend(loc='upper left')
    plt.show() """
   
    print("######### KNN Grafica genero #########")

    knn = KNeighborsClassifier(algorithm='brute',n_neighbors=80)
        
    knn.fit(X_train,y_train) 
    y_predic = knn.predict(X_test)

    print("y predict", y_predic)
    #get presision
    score = knn.score(X_train,y_train)
    print(score)
    #get matrix confision
    conf_matrix = confusion_matrix(y_test,y_predic)
    print(conf_matrix)

    y_test = np.array(y_test)
    #y_predic = np.array(y_predic)
    x_matrix_data =  np.array(X_test)[:,0]
    y_matrix_data =  np.array(X_test)[:,1]

    
    fig, ax = plt.subplots(2)
    plt.title("Naive Bayes")

    X_height_male = []
    Y_weight_male = []

    X_height_famele = []
    Y_weight_female = []


    for i in range(len(indices_test)):
        if (y_test[i]==1): #case male
            X_height_male.append(x_matrix_data[i])
            Y_weight_male.append(y_matrix_data[i])
        else: #case female
            X_height_famele.append(x_matrix_data[i])
            Y_weight_female.append(y_matrix_data[i])

    ax[0].scatter(X_height_male,Y_weight_male)
    ax[0].scatter(X_height_famele,Y_weight_female,color='red')
    ax[0].set_title('Test')
    ax[0].set_xlabel('Height')
    ax[0].set_ylabel('Weight')


    X_height_male_predict = []
    Y_weight_male_predic = []
    X_height_female_predict = []
    Y_weight_female_predic = []

    for i in range(len(indices_test)):
        if (y_predic[i]==1): #case male
            X_height_male_predict.append(x_matrix_data[i])
            Y_weight_male_predic.append(y_matrix_data[i])
        else: #case female
            X_height_female_predict.append(x_matrix_data[i])
            Y_weight_female_predic.append(y_matrix_data[i])


    ax[1].scatter(X_height_male_predict,Y_weight_male_predic)
    ax[1].scatter(X_height_female_predict,Y_weight_female_predic,color='red')
    ax[1].set_title('Predic')
    ax[1].set_xlabel('Height')
    ax[1].set_ylabel('Weight')


    plt.show()


    ##----------------------------- REGRESION LOGISTICA --------------------------
    print("###REGRESION LOGISTICA")
    print("DATA SET DEFAULT")
                                    #Data set default

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

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_predic = clf.predict(X_test)
    print("y predict", y_predic)
    #get presision
    score = clf.score(X_train,y_train)
    print(score)
    #get matrix confision
    conf_matrix = confusion_matrix(y_test,y_predic)
    print(conf_matrix)

                                        #Data set genero
    print("DATA SET GENERO")

    #Lectura de Datos
    data_default = pd.read_csv(filename2,sep=",",header=0)

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
    #X_train,X_test,y_train,y_test = train_test_split(X,y,indices,test_size=0.2)
    X_train,X_test,y_train,y_test,indices_train,indices_test = train_test_split(X, y,indices, test_size=0.2)
    
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_predic = clf.predict(X_test)
    print("y predict", y_predic)
    #get presision
    score = clf.score(X_train,y_train)
    print(score)
    #get matrix confision
    conf_matrix = confusion_matrix(y_test,y_predic)
    print(conf_matrix)

    y_test = np.array(y_test)
    #y_predic = np.array(y_predic)
    x_matrix_data =  np.array(X_test)[:,0]
    y_matrix_data =  np.array(X_test)[:,1]

    
    fig, ax = plt.subplots(2)
    plt.title("Naive Bayes")

    X_height_male = []
    Y_weight_male = []

    X_height_famele = []
    Y_weight_female = []


    for i in range(len(indices_test)):
        if (y_test[i]==1): #case male
            X_height_male.append(x_matrix_data[i])
            Y_weight_male.append(y_matrix_data[i])
        else: #case female
            X_height_famele.append(x_matrix_data[i])
            Y_weight_female.append(y_matrix_data[i])

    ax[0].scatter(X_height_male,Y_weight_male)
    ax[0].scatter(X_height_famele,Y_weight_female,color='red')
    ax[0].set_title('Test')
    ax[0].set_xlabel('Height')
    ax[0].set_ylabel('Weight')


    X_height_male_predict = []
    Y_weight_male_predic = []
    X_height_female_predict = []
    Y_weight_female_predic = []

    for i in range(len(indices_test)):
        if (y_predic[i]==1): #case male
            X_height_male_predict.append(x_matrix_data[i])
            Y_weight_male_predic.append(y_matrix_data[i])
        else: #case female
            X_height_female_predict.append(x_matrix_data[i])
            Y_weight_female_predic.append(y_matrix_data[i])


    ax[1].scatter(X_height_male_predict,Y_weight_male_predic)
    ax[1].scatter(X_height_female_predict,Y_weight_female_predic,color='red')
    ax[1].set_title('Predic')
    ax[1].set_xlabel('Height')
    ax[1].set_ylabel('Weight')


    plt.show()

    ##----------------------NAIVE BAYES ------------------------------
    #DATA SET GENERO
    #Lectura de Datos
    data_default = pd.read_csv(filename2,sep=",",header=0)

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

    y_test = np.array(y_test)
    #y_predic = np.array(y_predic)
    x_matrix_data =  np.array(X_test)[:,0]
    y_matrix_data =  np.array(X_test)[:,1]

    
    fig, ax = plt.subplots(2)
    plt.title("Naive Bayes")

    X_height_male = []
    Y_weight_male = []

    X_height_famele = []
    Y_weight_female = []


    for i in range(len(indices_test)):
        if (y_test[i]==1): #case male
            X_height_male.append(x_matrix_data[i])
            Y_weight_male.append(y_matrix_data[i])
        else: #case female
            X_height_famele.append(x_matrix_data[i])
            Y_weight_female.append(y_matrix_data[i])

    ax[0].scatter(X_height_male,Y_weight_male)
    ax[0].scatter(X_height_famele,Y_weight_female,color='red')
    ax[0].set_title('Test')
    ax[0].set_xlabel('Height')
    ax[0].set_ylabel('Weight')


    X_height_male_predict = []
    Y_weight_male_predic = []
    X_height_female_predict = []
    Y_weight_female_predic = []

    for i in range(len(indices_test)):
        if (y_predic[i]==1): #case male
            X_height_male_predict.append(x_matrix_data[i])
            Y_weight_male_predic.append(y_matrix_data[i])
        else: #case female
            X_height_female_predict.append(x_matrix_data[i])
            Y_weight_female_predic.append(y_matrix_data[i])


    ax[1].scatter(X_height_male_predict,Y_weight_male_predic)
    ax[1].scatter(X_height_female_predict,Y_weight_female_predic,color='red')
    ax[1].set_title('Predic')
    ax[1].set_xlabel('Height')
    ax[1].set_ylabel('Weight')


    plt.show()

main()