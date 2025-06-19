from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras', epsilon=None):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)
        # Podés guardar epsilon si querés, aunque no lo uses.
        self.epsilon = epsilon

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        """

        # Distancia horizontal al próximo tubo
        distancia_tubo = state["next_pipe_dist_to_player"] #lo que falta para llegar al tubo

        # Altura del centro del hueco del tubo
        centro_hueco = (state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) / 2 #el punto medio del hueco entre tubos

        # Diferencia vertical entre el pájaro y el centro del hueco
        diferencia_altura = state["player_y"] - centro_hueco # que tan arriba o abajo está el pájaro respecto del hueco

        # Velocidad actual del pájaro
        velocidad_pajaro = state["player_vel"] # que tan rapido sube o baja

        # Discretización: agrupamos valores en intervalos (binning)
        bin_distancia = int(distancia_tubo // 10)         # agrupar cada 10 píxeles
        bin_altura = int(diferencia_altura // 10)         # agrupar cada 10 píxeles
        bin_velocidad = int(velocidad_pajaro)             # ya es un número discreto (-9 a 10)

     # Representamos el estado como una tupla de valores discretos
        return (bin_distancia, bin_altura, bin_velocidad)

    def act(self, state):
        """
        COMPLETAR: Implementar la función de acción.
        Debe transformar el estado al formato de entrada de la red y devolver la acción con mayor Q.
        """
        # Discretizamos el estado
        discrete_state = self.discretize_state(state)

        # Política epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            # Convertimos a array para la red
            state_array = np.array(discrete_state, dtype=np.float32).reshape(1, -1)
            q_values = self.model.predict(state_array, verbose=0)[0]
            action_index = np.argmax(q_values)
            return self.actions[action_index]
