import numpy as np

class FuzzyCognitiveMap:
    def __init__(self, concepts, weights, threshold=0.01, max_iter=100, decay=0.5):
        self.concepts = concepts
        self.weights = weights
        self.threshold = threshold
        self.max_iter = max_iter
        self.decay = decay

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def run(self, input_vector):
        state = np.array(input_vector, dtype=float)
        for _ in range(self.max_iter):
            next_state = self.sigmoid(np.dot(state, self.weights)) * self.decay + state * (1 - self.decay)
            diff = np.linalg.norm(next_state - state)
            state = next_state
            if diff < self.threshold:
                break
        return state

def fuzzy_predict(X):
    # Índices de columnas relevantes para FCM (C1, C18, C19, C6, C20)
    indices = [0, 17, 18, 5, 19]

    # Matriz de pesos para influencias entre conceptos
    weights = np.array([
        [0,    0.3,  0.2,  0,    0  ],
        [0,    0,    0.4,  0.1,  0  ],
        [0,    0,    0,    0.5,  0  ],
        [0.2,  0,    0,    0,    0.4],
        [0,    0,    0,    0,    0   ],
    ])

    predictions = []
    fcm = FuzzyCognitiveMap(concepts=5, weights=weights)

    for x in X:
        input_vec = x[indices]
        # Normalizamos las variables según sus máximos teóricos
        norm_vec = np.array([
            input_vec[0] / 50,    # Edad (C1)
            input_vec[1] / 250,   # Presión sistólica (C18)
            input_vec[2] / 150,   # Presión diastólica (C19)
            input_vec[3] / 2,     # Síntoma inicial (C6)
            input_vec[4] / 5      # Motivo de parto (C20)
        ])

        output = fcm.run(norm_vec)
        risk_score = output[-1]
        pred = 1 if risk_score > 0.5 else 0
        predictions.append(pred)

    return np.array(predictions)
