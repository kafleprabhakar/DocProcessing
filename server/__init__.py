from flask import Flask
import os

server_root = os.path.dirname(__file__)
template_path = os.path.abspath(os.path.join(server_root, 'templates'))
static_path = os.path.abspath(os.path.join(server_root, 'static'))

flask_app = Flask(__name__, template_folder=template_path, static_folder=static_path)