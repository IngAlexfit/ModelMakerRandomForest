import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from joblib import dump
# Carga el dataset CSV en un DataFrame de Pandas
df = pd.read_csv('dataPruebasSaberPro.csv')

# Separa las características (X) y la clase objetivo (y)
X = df.drop('Class', axis=1)
y = df['Class']
print("Fin Class")
# Divide el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Fin Divi")
# Define los hiperparámetros a probar
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
print("Fin param_grid")
# Creamos el modelo Inicial  RamdomForestClassifier :)
clf = RandomForestClassifier()
print("Fin Model Inicial")
# GridSearchCV ->  Aplicamos la Validacion Cruzada con 5 folds Xd
grid_search = GridSearchCV(clf, param_grid, cv=5)
print("Fin Validacion Cruzada")

grid_search.fit(X_train, y_train)

# Muestra los mejores hiperparámetros encontrados (los que dieron mejor rendimiemto en la validacion cruzada)
print(f'Mejores hiperparámetros: {grid_search.best_params_}')

# Calcula el porcentaje de precisión del modelo en el conjunto de prueba
accuracy = grid_search.score(X_test, y_test) * 100

print(f'Precisión: {accuracy:.2f}%')

# Prompt Prueba = COLOMBIA,ANTIOQUIA,ANTIOQUIA,PUBLICIDAD,PRESENCIAL,Entre 1 millón y menos de 2.5 millones,ANTIOQUIA,No,No,Entre 11 y 20 horas,N,COLOMBIA,M,Si,No,Secundaria (Bachillerato) completa,No,Si,Estrato 3,Si,Si,Secundaria (Bachillerato) completa,NO OFICIAL - FUNDACIÓN,Por encima
new_data = [[31,32,19,255,1,2,15,0,0,4,0,31,1,1,1,3,0,0,3,0,0,6,3]]
prediction = grid_search.predict(new_data)


print(f'Predicción: {prediction[0]}')
# Guardamos en modelo en un archivo
dump(grid_search.best_estimator_, 'modelopruebaforest8.joblib')