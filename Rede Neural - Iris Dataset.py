# ## Rede Neural com Normalização de Dados - Iris Dataset

import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(5) #Fixando os valores.

scaler = StandardScaler() #Adicionando o StandardScaler para normalização dos dados.

iris = datasets.load_iris() #Usando o dataset Iris.

X = iris.data #Dados.
y = iris.target  #Rótulos.

scaler.fit(X)
X = scaler.transform(X)

sufInd = np.arange(150) #Números aleatórios de 0 a 149.
np.random.shuffle(sufInd) #Mistura os números.

X_train = X[sufInd[:100],:] #Pega as 100 primeiras amostras do dataset e usa para treinar.
X_test = X[sufInd[100:],:] #Pega as 50 últimas amostras do dataset e usa para teste. :] = todas as colunas.

y_train = y[sufInd[:100]]
y_test = y[sufInd[100:]]

classifier = MLPClassifier(solver ='sgd', hidden_layer_sizes=(3), learning_rate_init = 0.01, activation='logistic', max_iter=1500, random_state=1)#learning_rate_init=0.01 equevale a 1 para 100.
classifier.fit(X_train, y_train) #Fit - Vai fazer o treinamento.
y_prediction = classifier.predict_proba(X_test)#Faz a predição e probabilidade.
y_aux = np.argmax(y_prediction, 1)
accuracy_score(y_test, y_aux)
#print(y_prediction) #Sai com 3 probabilidades em cada linha de dados

y_prediction

print(y_test[:10], y_aux[:10])

X_train


#Conclusões

# - Antes da normalização, a rede neural obteve apenas 28% de accuracy. Após a normalização dos dados, obteve 86%.
# - Com o learning_rate_init saindo de 0.001 para 0.01, se obteve uma accuracy de 98% (modificações mais radicais mantém o mesmo valor)
# - A normalização foi feita entre os valores de -1 e 1.
# - Taxa de variação dos valores na mesma faixa, com isso a rede neural pôde otimizar o aprendizado.
