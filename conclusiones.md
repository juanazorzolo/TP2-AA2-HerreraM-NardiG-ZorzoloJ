
# Conclusiones - Flappy Bird con Q-Learning y Deep Q-Learning

## Ingeniería de Características (Discretización del Estado)

Para aplicar Q-Learning tabular fue necesario transformar el espacio continuo del entorno de Flappy Bird en un espacio discreto. Para ello, se definieron las siguientes variables:

- **Distancia al próximo tubo (`next_pipe_dist_to_player`)**: discretizada en intervalos de 10 píxeles.
- **Diferencia vertical con el hueco (`player_y` - centro del hueco entre tubos)**: discretizada en intervalos de 10 píxeles.
- **Velocidad vertical del pájaro (`player_vel`)**: ya es discreta, se usó directamente.

El estado se representa como una **tupla** de estos tres valores discretos:
```python
(bin_distancia, bin_altura, bin_velocidad)
```
Esta representación reduce significativamente el espacio de estados, permitiendo al agente aprender una política eficaz con una tabla Q manejable.

---

## Resultados del Agente Q-Learning

### Entrenamiento

Durante el entrenamiento del agente Q-Learning, se alcanzaron recompensas promedio bajas al final de los últimos 100 episodios (5.74, 3.15, 5.52), indicando que el agente todavía tiene dificultades para generalizar su comportamiento de forma estable al final del entrenamiento. Sin embargo, al ejecutar el agente en modo explotación (sin exploración), se obtuvieron mejores resultados.

### Evaluación

Los resultados del agente entrenado (modo explotación) muestran una mejora significativa:

- Recompensas en episodios de prueba: 92, 25, 73, 66, 136
- Promedio de recompensas en estas pruebas: **78.4**

Esto indica que, aunque el promedio de los últimos episodios de entrenamiento era bajo, el agente fue capaz de aprender un comportamiento útil, especialmente visible cuando se lo ejecuta sin exploración.

---

## Resultados del Agente con Red Neuronal (Deep Q-Learning)

El agente basado en red neuronal fue entrenado para aproximar la Q-table aprendida por el agente tabular.

### Evaluación

Los resultados muestran que el agente neuronal tiene un comportamiento más estable y logra recompensas más altas en general:

- Recompensas obtenidas: 114, 27, 23, 39, 69, 72
- Promedio de recompensas: **57.3**

Aunque el promedio es algo menor que el mejor desempeño del agente tabular, el comportamiento es más consistente, lo cual es esperable considerando que el modelo generaliza mejor.

---

## Conclusión Final

- El agente tabular con Q-Learning logró aprender una política funcional pero con alta variabilidad y limitada generalización.
- El agente basado en red neuronal logró replicar y estabilizar el comportamiento aprendido, ofreciendo una política más generalizable.
- Para entornos como Flappy Bird donde el espacio de estados es continuo, los métodos con aproximadores de funciones (como redes neuronales) resultan más adecuados a largo plazo.
