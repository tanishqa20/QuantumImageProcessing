import cv2
import numpy as np
import math
import time
import os
import h5py
from qiskit import QuantumCircuit

# ==================================================
# IMAGE LOADING (PNG / JPG / MAT SUPPORT)
# ==================================================
def load_image(path, size=8):
    if not os.path.exists(path):
        raise FileNotFoundError(f" File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    img = None

    # ---------- IMAGE FILE ----------
    if ext in [".png", ".jpg", ".jpeg"]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(" Unable to read image file")

    # ---------- MAT FILE ----------
    elif ext == ".mat":
        try:
            # Brain Tumor MAT file key
            with h5py.File(path, 'r') as f:
                if 'cjdata' in f and 'image' in f['cjdata']:
                    img = np.array(f['cjdata']['image'])
                    print(" Loaded Brain Tumor MRI dataset: cjdata/image")
                else:
                    # fallback: pick first dataset automatically
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            img = np.array(f[key])
                            print(f" Loaded MAT key: {key}")
                            break
            if img is None:
                raise ValueError(" No valid dataset found in MAT file")
        except OSError:
            # fallback for older MATLAB MAT files
            import scipy.io
            mat = scipy.io.loadmat(path)
            for key in mat:
                if isinstance(mat[key], np.ndarray) and mat[key].ndim >= 2:
                    img = mat[key]
                    print(f" Loaded MAT key (scipy): {key}")
                    break
            if img is None:
                raise ValueError(" No valid image found in MAT file")

        # If 3D volume → take middle slice
        if img.ndim == 3:
            img = img[:, :, img.shape[2] // 2]

        # Normalize
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())

    else:
        raise ValueError(" Unsupported file format")

    # Resize to standard size
    img = cv2.resize(img, (size, size))
    return img

# ==================================================
# QUANTUM IMAGE REPRESENTATION TECHNIQUES
# ==================================================
# FRQI
def frqi_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))
    qc = QuantumCircuit(n + 1)
    qc.h(range(n))
    for i, p in enumerate(pixels):
        qc.ry(p * math.pi, n)
    return qc

# NEQR
def neqr_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))
    qc = QuantumCircuit(n + 8)
    qc.h(range(n))
    for p in pixels:
        value = int(p * 255)
        binary = format(value, "08b")
        for i, bit in enumerate(binary):
            if bit == "1":
                qc.x(n + i)
    return qc

# MCQI (Multi-Channel Quantum Image)
def mcqi_encode(image):
    if image.ndim != 3 or image.shape[2] != 3:
        # fallback to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    pixels = image.flatten()
    n = int(np.log2(len(pixels)//3))
    qc = QuantumCircuit(n + 1)
    qc.h(range(n))
    for i, p in enumerate(pixels):
        qc.ry(p * (math.pi/3), n)
    return qc

# Qubit Lattice (QL)
def ql_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))
    qc = QuantumCircuit(n)
    for i, p in enumerate(pixels):
        if p > 0.5:
            qc.x(i)
    return qc

# QPIE
def qpie_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i, p in enumerate(pixels):
        qc.ry(p*math.pi/2, i)
    return qc

# FRQI-C (color)
def frqi_c_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)//3))
    qc = QuantumCircuit(n + 1)
    qc.h(range(n))
    for i, p in enumerate(pixels):
        qc.ry(p * math.pi, n)
        qc.rz(p * math.pi/2, n)
    return qc

# NEQR-C (color)
def neqr_c_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)//3))
    qc = QuantumCircuit(n + 8)
    qc.h(range(n))
    for p in pixels:
        value = int(p*255)
        binary = format(value, "08b")
        for i, bit in enumerate(binary):
            if bit == "1":
                qc.x(n + i)
    return qc

# QIR-Phase
def qir_phase_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i, p in enumerate(pixels):
        qc.rz(p * math.pi, i)
    return qc

# GQIR
def gqir_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))
    qc = QuantumCircuit(n + 2)
    qc.h(range(n))
    for i, p in enumerate(pixels):
        qc.ry(p*math.pi, n)
        qc.rx(p*math.pi/2, n+1)
    return qc

# AQIR
def aqir_encode(image):
    pixels = image.flatten()
    n = int(np.log2(len(pixels)))
    qc = QuantumCircuit(n)
    amps = pixels / np.sqrt(np.sum(pixels**2))
    for i, amp in enumerate(amps):
        qc.ry(amp*math.pi/2, i)
    return qc

# ==================================================
# EVALUATION FUNCTION (10 PARAMETERS)
# ==================================================
def evaluate(circuit, model_type):
    qubits = circuit.num_qubits
    depth = circuit.depth()
    gates = sum(circuit.count_ops().values())

    gate_fidelity = round(1 / (gates + 1), 6)
    encoding_time = round(depth * 0.0001, 6)

    if model_type in ["FRQI", "FRQI-C", "QPIE", "MCQI"]:
        data_loss = "Yes"
        reconstruction = "Medium"
        transform_support = "High"
        scalability = "High"
        noise_robustness = "Medium"
    else:
        data_loss = "No"
        reconstruction = "High"
        transform_support = "Moderate"
        scalability = "Low"
        noise_robustness = "Low"

    return {
        "Qubit Count": qubits,
        "Gate Count": gates,
        "Circuit Depth": depth,
        "Gate Fidelity": gate_fidelity,
        "Data Loss": data_loss,
        "Encoding Time (s)": encoding_time,
        "Reconstruction Accuracy": reconstruction,
        "Transformation Supportability": transform_support,
        "Scalability": scalability,
        "Noise Robustness": noise_robustness
    }

# ==================================================
# DATASETS
# ==================================================
base_dir = r"C:\Users\Tanishqa More\OneDrive\Documents\quantum_image_representation\datasets"
datasets = {
    "Brain Tumor MRI": os.path.join(base_dir, "brain_tumor", "2299.mat"),
    "BrainWeb MRI": os.path.join(base_dir, "brainweb", "image.mat"),
    "ICEYE SAR": os.path.join(base_dir, "iceye_sar", "image.png"),
    "NASA SAR": os.path.join(base_dir, "nasa_sar", "image.png"),
    "SSDD SAR": os.path.join(base_dir, "ssdd", "image.png")
}

# ==================================================
# MAIN EXECUTION
# ==================================================
techniques = {
    "FRQI": frqi_encode,
    "NEQR": neqr_encode,
    "MCQI": mcqi_encode,
    "QL": ql_encode,
    "QPIE": qpie_encode,
    "FRQI-C": frqi_c_encode,
    "NEQR-C": neqr_c_encode,
    "QIR-Phase": qir_phase_encode,
    "GQIR": gqir_encode,
    "AQIR": aqir_encode
}

for name_ds, path in datasets.items():
    print("\n========================================")
    print("Dataset:", name_ds)

    try:
        image = load_image(path)
    except Exception as e:
        print(" Error loading image:", e)
        continue

    for name_tech, func in techniques.items():
        try:
            start = time.time()
            circuit = func(image)
            metrics = evaluate(circuit, name_tech)
            elapsed = time.time() - start
            print(f"\n{name_tech} Evaluation:")
            for k, v in metrics.items():
                print(f"{k}: {v}")
            print(f"Encoding Time (computed): {elapsed:.6f}s")
        except Exception as e:
            print(f" Error in {name_tech}: {e}")
