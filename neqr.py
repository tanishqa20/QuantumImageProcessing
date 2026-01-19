import numpy as np
from qiskit import QuantumCircuit

def neqr_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))

    qc = QuantumCircuit(n + 8)
    qc.h(range(n))

    for pixel in pixels:
        binary = format(int(pixel * 255), '08b')
        for i, bit in enumerate(binary):
            if bit == '1':
                qc.x(n + i)

    return qc
