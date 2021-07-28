import json
from base.classes import Box, Checkbox, Cell
import numpy as np

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Box, Checkbox, Cell)):
            return obj._to_json()
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return json.JSONEncoder.default(self, obj)