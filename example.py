# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Paso 1: Cargar el dataset de Iris
iris = load_iris()
X = iris.data  # Características (features)
y = iris.target  # Etiquetas (labels)

# Paso 2: Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Paso 3: Escalar las características (opcional pero recomendado para KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Paso 4: Crear el clasificador KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Paso 5: Entrenar el clasificador
knn.fit(X_train, y_train)

# Paso 6: Hacer predicciones
y_pred = knn.predict(X_test)

# Paso 7: Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Paso 8: Graficar (opcional, solo para dos características)
# Colores distintos por clase
plt.figure(figsize=(8, 6))

# Graficamos los puntos de prueba según las predicciones y colores por clase
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k', s=100, alpha=0.7)

# Añadir título y etiquetas
plt.title('Predicciones del clasificador KNN')
plt.xlabel('Característica 1 (longitud del sépalo)')
plt.ylabel('Característica 2 (ancho del sépalo)')

# Añadir una leyenda para identificar las clases
plt.legend(*scatter.legend_elements(), title="Clases")

# Mostrar el gráfico
plt.show()
