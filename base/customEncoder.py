import json
from base.classes import Box
import numpy as np

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        print('json: ', obj)
        if isinstance(obj, Box):
            return obj.get_vertices().tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return json.JSONEncoder.default(self, obj)