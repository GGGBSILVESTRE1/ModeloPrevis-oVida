import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Carregar o dataset
dataset = pd.read_excel('expec_vida.xlsx')

# Função para limpar e converter colunas de string para float
def clean_and_convert(column):
    return column.str.replace(',', '').astype(float)

# Limpar a coluna 3 
dataset.iloc[:, 3] = clean_and_convert(dataset.iloc[:, 3])

#divisão dos dataset
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]

# Divisão para treino do modelo
X_train = X.iloc[:44, :]
y_train = y.iloc[:44]

# Divisão para o teste, com os 5 países separados 
X_test = X.iloc[44:50, :]
y_test = y.iloc[44:50]

# Métoddo IQR para limpar os outlayers que afetam a previssibilidade do modelo
def remove_outliers_iqr(X, y):
    combined = pd.concat([X, y], axis=1)
    Q1 = combined.quantile(0.35)
    Q3 = combined.quantile(0.65)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    no_outliers = combined[(combined >= lower_bound) & (combined <= upper_bound)].dropna()
    return no_outliers.iloc[:, :-1], no_outliers.iloc[:, -1]

X_train, y_train = remove_outliers_iqr(X_train, y_train)

# Padronizar os dados para melhor otimização do modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Treinar o modelo de regressão linear
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

# Fazer previsões com o conjunto de teste, 
# segue o cálculo de X_test_scaled x coef, chegando no valor y_pred
y_pred = regressor.predict(X_test_scaled)

# Calcular o erro ao quadrado médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Comparar previsões com valores reais em colunas
y_compare = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), axis=1)
print(y_compare)

# Imprimir os coeficientes e a intercepção da fórmula com eixo y, ou seja o valor quando as variaǘeis são 0
coef = regressor.coef_
intercept = regressor.intercept_

print('Coeficientes:', coef)
print('Intercepto:', intercept)

# Fórmula do modelo, onde X1,X2,X3 assume os valores padronizados no X_test_scaled
print("Fórmula do modelo:")
formula = "Expectativa de Vida = {:.2f}".format(intercept)
for i, c in enumerate(coef):
    formula += " + ({:.2f} * X{})".format(c, i + 1)
print(formula)

print("matriz de coeficientes para cálculo:" )
print(X_test_scaled)

# Observação gráfica da aproximação, onde o eixo x representa os valores reais
#  e o eixo y os valores da predição, e a reta o caso onde x=y para comparação
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Linha de referência')
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs. Valores Previstos')
plt.show()




