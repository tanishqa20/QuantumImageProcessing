import numpy as np
import math
from qiskit import QuantumCircuit

def frqi_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))

    qc = QuantumCircuit(n + 1)
    qc.h(range(n))

    for pixel in pixels:
        theta = pixel * math.pi
        qc.ry(theta, n)

    return qc
