with open("models/final/modelo_gestos.tflite", "rb") as f:
    tflite_data = f.read()

with open("models/final/modelo_gestos.h", "w") as f:
    f.write("const unsigned char modelo_gestos[] = {\n  ")
    f.write(", ".join(f"0x{byte:02x}" for byte in tflite_data))
    f.write(f"\n}};\nconst unsigned int modelo_gestos_len = {len(tflite_data)};\n")
