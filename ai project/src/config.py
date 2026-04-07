import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"

# File to store registered person metadata (ID, name, etc.)
METADATA_FILE = DATA_DIR / "people.json"

# Camera index for OpenCV (0 is default webcam)
CAMERA_INDEX = 0
