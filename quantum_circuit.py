
import pennylane as qml
import numpy as np

def quantum_circuit(features, params):
    n_qubits = len(features)
    qml.templates.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)
    for i in range(n_qubits):
        qml.Rot(*params[i], wires=i)
    return qml.expval(qml.PauliZ(0))

def quantum_cnn(features, n_qubits, n_layers=3):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def qnode(params):
        return quantum_circuit(features, params)
    params = np.random.uniform(0, np.pi, (n_layers, n_qubits, 3))
    results = [qnode(params[layer]) for layer in range(n_layers)]
    return np.mean(results)
