from preprocess import load_and_preprocess
from frqi import frqi_encode
from neqr import neqr_encode
from qiskit import Aer, execute

datasets = {
    "Brain Tumor MRI": "datasets/brain_tumor/image.png",
    "BrainWeb": "datasets/brainweb/image.png",
    "ICEYE SAR": "datasets/iceye_sar/image.png",
    "NASA SAR": "datasets/nasa_sar/image.png",
    "SSDD": "datasets/ssdd/image.png"
}

backend = Aer.get_backend("statevector_simulator")

for name, path in datasets.items():
    print(f"\nDataset: {name}")

    image = load_and_preprocess(path)

    frqi_circuit = frqi_encode(image)
    neqr_circuit = neqr_encode(image)

    execute(frqi_circuit, backend).result()
    execute(neqr_circuit, backend).result()

    print("FRQI → Qubits:", frqi_circuit.num_qubits,
          "Depth:", frqi_circuit.depth(),
          "Gates:", sum(frqi_circuit.count_ops().values()))

    print("NEQR → Qubits:", neqr_circuit.num_qubits,
          "Depth:", neqr_circuit.depth(),
          "Gates:", sum(neqr_circuit.count_ops().values()))
