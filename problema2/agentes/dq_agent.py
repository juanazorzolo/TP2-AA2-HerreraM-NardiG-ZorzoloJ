############## MÉTODOS A IMPLEMENTAR PARA EL TP:  ########################

""" 1. Método: discretize_state(self, state)
    Este método debe convertir el estado continuo (con valores numéricos que te da el juego)
      en una tupla de valores discretos para poder indexar la Q-table.

    EL ESTADO QUE RECIBIMOS ES UN DICCIONARIO COMO ->  {'player_y': 256,'player_vel': 0,.....} 

    Ejemplo básico de discretización:

    Dividir o agrupar valores continuos en bins (intervalos).

    Por ejemplo: player_y entre 0-50 → bin 0, 51-100 → bin 1, etc.
""" 

"""
    2. Método act(self, state)
    Implementar la política epsilon-greedy:

    Con probabilidad epsilon, elegir acción al azar (explorar).

    Con probabilidad 1 - epsilon, elegir la acción con mayor Q-value para el estado discretizado (explotar).  
"""


from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle

class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        # TODO: Definir parámetros de discretización según el entorno

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
        Elige una acción usando epsilon-greedy sobre la Q-table.
        """

        """ QUÉ HACE -> Si un número aleatorio es menor a epsilon, elige una acción aleatoria (exploración).
        Si no, elige la acción con mayor valor Q para el estado actual (explotación).
        self.actions es la lista de acciones posibles (por ejemplo, ["flap", None])."""

        # Discretizamos el estado actual
        estado_discreto = self.discretize_state(state)

        # Política epsilon-greedy:
        if np.random.rand() < self.epsilon:
            # Explorar: elegir una acción al azar
            return np.random.choice(self.actions)
        else:
            # Explotar: elegir la mejor acción según la Q-table
            q_values = self.q_table[estado_discreto]
            mejor_indice = np.argmax(q_values)
            return self.actions[mejor_indice]
    

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
