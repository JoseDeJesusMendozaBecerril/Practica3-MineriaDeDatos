#Librerias
from sklearn import datasets
from sklearn.model_selection import train_test_split #separa data
from sklearn.preprocessing import StandardScaler #Escalamiento
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score #metricas precision
from sklearn.metrics import recall_score #sensibilidad
from sklearn.metrics import roc_auc_score

#Importamos el data set
dataset = datasets.load_breast_cancer()
#print(dataset)
#print(dataset.keys())

#Seleccionamos todas lascolumnas
X = dataset.data

#Seleccionamos datos de salida
y = dataset.target


#Separo datos
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#Escalar datos
escalar = StandardScaler()
X_train = escalar.fit_transform(X_train)
X_test = escalar.transform(X_test)

#Defino el algoritmo
algoritmo = LogisticRegression()
algoritmo.fit(X_train,y_train)


#Realizo prediccion
y_pred = algoritmo.predict(X_test)

#Verifico matriz de confusion
matriz = confusion_matrix(y_test,y_pred)

print("y_test",y_test)
print("y_pred",y_pred)

print("Matriz de confusion")
print(matriz)

#Calculo precision del algoritmo
precision = precision_score(y_test,y_pred,average='binary')
print("Precision del modelo")
print(precision)

#Calculo de curva ROC - AUC del modelo
roc_auc = roc_auc_score(y_test,y_pred)
print("Curva ROC - AUC del modelo")
print(roc_auc)




