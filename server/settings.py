import os
# -----------------------------------------------------------------------------
# Settings related to server
# -----------------------------------------------------------------------------
SERVER_HOST = 'localhost'
SERVER_PORT = 2000

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, 'upload/')
OUTPUT_FOLDER = os.path.join(CURRENT_DIR, 'static', 'outputs/')
TEMPLATE_FOLDER = os.path.join(ROOT_DIR, 'template/')