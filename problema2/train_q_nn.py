import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table.pkl'  # Cambia el path si es necesario
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)
X = np.array(X)
y = np.array(y)

# --- Definir la red neuronal ---
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),  # tamaño del estado
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(y.shape[1])  # salida con tantas neuronas como acciones posibles
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- Entrenar la red neuronal ---
# COMPLETAR: Ajustar hiperparámetros según sea necesario
# model.fit(X, y, ... demas opciones de entrenamiento ...)
history = model.fit(X, y, epochs=70, batch_size=32, verbose=1)

# --- Mostrar resultados del entrenamiento ---
# Completar: Imprimir métricas de entrenamiento
plt.plot(history.history['loss'], label='MSE (Loss)')
plt.plot(history.history['mae'], label='MAE')
plt.title('Pérdida y Error Absoluto durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.legend()
plt.grid()
plt.show()

# --- Guardar el modelo entrenado ---
# COMPLETAR: Cambia el nombre si lo deseas
model.save('flappy_q_nn_model.keras')
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
