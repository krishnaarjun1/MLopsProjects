import os

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
