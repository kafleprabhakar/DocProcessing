import sys
import os
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
sys.path.append(CURRENT_DIR)
sys.path.append(CURRENT_DIR+'/services/')
sys.path.append(CURRENT_DIR+'/server/')

from server import flask_app
from server import views
from server.settings import SERVER_HOST, SERVER_PORT

# start flask service
if __name__ == "__main__":
    flask_app.run(host=SERVER_HOST, port=SERVER_PORT, debug=True, threaded=True)