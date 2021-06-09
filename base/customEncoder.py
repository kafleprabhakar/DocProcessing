import json
from base.box import Box

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        print('json: ', obj)
        if isinstance(obj, Box):
            return obj.get_vertices().tolist()
        else:
            return json.JSONEncoder.default(self, obj)