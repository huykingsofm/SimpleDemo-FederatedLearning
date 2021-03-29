import os
import threading

from main import Listener
from main import MODEL

SERVER_DIR = "Server"
MODEL_FILE_NAME = os.path.join(SERVER_DIR, "SimpleNeuronNetwork01.weight")

if __name__ == "__main__":
    if not os.path.isdir(SERVER_DIR):
        os.mkdir(SERVER_DIR)

    model = MODEL()
    if not model.checkVersion(MODEL_FILE_NAME):
        model.write(MODEL_FILE_NAME)

    listener = Listener(
        server_address= ("127.0.0.1", 1999),
        architecture= MODEL, 
        directory= SERVER_DIR,
        verbosities={
        "user": ["notification", "error"],
        "dev": ["error"]
    })
    listener_thread = threading.Thread(target=listener.start)
    listener_thread.start()
