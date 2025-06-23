# Trabajo Práctico 2 - Aprendizaje Automático II

Autores: Herrera Morena, Nardi Gianella Belén, Zorzolo Rubio Juana   
Carrera: Tecnicatura Universitaria en Inteligencia Artificial - FCEIA UNR  
Año: 2025


---

## Problema 1 - Clasificación de Dígitos con Audio (Audio MNIST)

### Descripción
Este problema aborda la tarea de clasificación de dígitos hablados del 0 al 9 utilizando un conjunto de datos de audio (`spoken_digit`) disponible en TensorFlow Datasets. El dataset contiene un total de 2500 clips de audio provenientes de 5 locutores (50 grabaciones por dígito por locutor).

### Objetivo
Construir un modelo de clasificación de dígitos hablados a partir de espectrogramas generados desde los clips de audio. Se entrenan y evalúan dos arquitecturas de redes neuronales:

- 🧠 **Modelo convolucional (CNN)** sobre espectrogramas.
- 🔁 **Modelo recurrente (RNN)** sobre espectrogramas.

### Entrega
- Notebook en Google Colab con:
  - Preprocesamiento y análisis exploratorio del dataset.
  - Extracción de espectrogramas.
  - Implementación de ambos modelos.
  - Métricas de evaluación (accuracy, matriz de confusión, curvas de pérdida, etc.).
  - Comparación entre modelos.

---

## Problema 2 - Agentes para Flappy Bird con Q-Learning

### Descripción
Se entrena un agente para jugar al juego Flappy Bird utilizando dos enfoques basados en Q-Learning. El entorno se construye con la librería PyGame Learning Environment (PLE).

### Ejercicio A: Agente Q-Learning Tabular

- 🧩 **Implementación**: Se completa el archivo `agentes/dq_agent.py`, discretizando el estado del juego y utilizando una Q-table.
- 🏋️‍♂️ **Entrenamiento**: `train_q_agent.py` genera la Q-table y la guarda en un archivo `.pkl`.
- 🎮 **Prueba**: El agente se prueba con `test_agent.py --agent agentes.dq_agent.QAgent`.

### Ejercicio B: Agente con Red Neuronal

- 🧠 **Entrenamiento de red**: Se entrena una red neuronal (`train_q_nn.py`) que aprende a aproximar la Q-table obtenida anteriormente.
- 🤖 **Agente neuronal**: Se completa `agentes/nn_agent.py` para que el agente use la red para decidir acciones.
- 🎮 **Prueba**: `test_agent.py --agent agentes.nn_agent.NNAgent`.

### Análisis Final

El archivo `problema2/conclusiones.md` contiene:
- Ingeniería de características (discretización del estado).
- Comparación de resultados entre el agente tabular y el basado en red neuronal.

---

## Estructura del Proyecto

TP2-AA2/
├── problema1/ # Notebook de clasificación de audio

├── problema2/

│ ├── agentes/

│ │ ├── base.py

│ │ ├── dq_agent.py

│ │ ├── nn_agent.py

│ │ ├── random_agent.py

│ │ └── manual_agent.py

│ ├── train_q_agent.py

│ ├── train_q_nn.py

│ ├── test_agent.py

│ ├── flappy_birds_q_table.pkl

│ ├── flappy_q_nn_model/ # Modelo SavedModel de TensorFlow

│ └── conclusiones.md

├── requirements.txt

└── README.md

---

## Requisitos

Para ejecutar localmente el problema 2, se requiere:

- Python 3.8+
- Virtualenv

Instalación:

```bash
python3 -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
pip install -r requirements.txt
```
---

## Instrucciones de Uso

### Ejecutar agentes en Flappy Bird

- Agente aleatorio:

```bash
python test_agent.py --agent agentes.random_agent.RandomAgent
```
- Agente manual:

```bash
python test_agent.py --agent agentes.manual_agent.ManualAgent
```

- Agente Q-learning entrenado:
```bash
python test_agent.py --agent agentes.dq_agent.QAgent
```

- Agente neuronal:
```bash
python test_agent.py --agent agentes.nn_agent.NNAgent
```

### Entrenamiento de Agentes

Entrenar agente Q-learning:

```bash
python train_q_agent.py
```
Entrenar red neuronal para aproximar la Q-table:

```bash
python train_q_nn.py
```
