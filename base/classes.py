import numpy as np
import copy
from typing import List, Tuple

class Box:
    """ Represents the immutable bounding box of the polygon passed """
    def __init__(self, vertices):
        """
        vertices: list of vertices either as [x1, y1, x2, y2, ..] or [[x1,y1], [x2,y2], ..]
                    or even [[[x1,y1]], [[x2,y2]], ...] or any higher dimension
        """
        vertices = np.array(vertices)
        flat_vertices = vertices.ravel()
        self.X = np.array([flat_vertices[i] for i in range(len(flat_vertices)) if i%2 == 0])
        self.Y = np.array([flat_vertices[i] for i in range(len(flat_vertices)) if i%2 == 1])
        self.vertices = np.array(list(zip(self.X, self.Y)))
    
    def get_X_range(self):
        return np.array([self.X.min(), self.X.max()])

    def get_Y_range(self):
        return np.array([self.Y.min(), self.Y.max()])
    
    def get_height(self):
        return self.Y.max() - self.Y.min()

    def get_width(self):
        return self.X.max() - self.X.min()

    def get_area(self):
        return self.get_height() * self.get_width()

    def get_center(self):
        return np.array([self.get_X_range().mean(), self.get_Y_range().mean()])
    
    def get_vertices(self):
        return copy.deepcopy(self.vertices)
    
    def get_box_endpoints(self):
        top_left, bottom_right = list(zip(self.get_X_range(), self.get_Y_range()))
        return top_left, bottom_right

    def check_overlap(self, other):
        self_X = self.get_X_range()
        self_Y = self.get_Y_range()

        other_X = other.get_X_range()
        other_Y = other.get_Y_range()

        return not (self_X[0] > other_X[1] or other_X[0] > self_X[1]) and not (self_Y[0] > other_Y[1] or other_Y[0] > self_Y[1])
    
    def contains(self, other):
        self_X = self.get_X_range()
        self_Y = self.get_Y_range()

        other_X = other.get_X_range()
        other_Y = other.get_Y_range()

        return self_X[0] <= other_X[0] and self_Y[0] <= other_Y[0] and self_X[1] >= other_X[1] and self_Y[1] >= other_Y[1]

    def check_duplicate(self, other, threshold: Tuple[int, int] = (10, 10)) -> bool:
        self_bounds = np.append(self.get_X_range(), self.get_Y_range())
        other_bounds = np.append(other.get_X_range(), other.get_Y_range())

        return all([abs(self_bounds[i] - other_bounds[i]) < threshold[i//2] for i in range(4)])

    def _to_json(self):
        return self.get_vertices().tolist()

    def __str__(self):
        return str(self.vertices.tolist())

    def __repr__(self):
        return f'Box({str(self)})'

    def __eq__(self, other):
        return np.allclose(self.get_X_range(), other.get_X_range()) and np.allclose(self.get_Y_range(), other.get_Y_range())

    def __hash__(self):
        return hash(str(self.vertices))


class Checkbox:
    """ Represents a checkbox in the document """
    def __init__(self, box: Box, percent_filled: float = None, label: str = None, patch: Box = None):
        self.box = box
        self.percent_filled = percent_filled
        self.label = label
        self.patch = patch
    
    def set_percent_filled(self, percent_filled: float) -> None:
        self.percent_filled = percent_filled
    
    def set_label(self, label: str) -> None:
        self.label = label
    
    def set_patch(self, patch: Box) -> None:
        self.patch = patch
    
    def get_box(self) -> Box:
        return self.box
    
    def get_patch_box(self) -> Box:
        return self.patch

    def _to_json(self):
        return {
            'box': self.box,
            'label': self.label,
            'percent_filled': self.percent_filled
        }

    def __str__(self):
        return str(self._to_json())

    def __repr__(self):
        return f'Checkbox({self.box}, label={self.label})'


class Cell:
    """ 
    Represents a table cell 
        boxes: list of boxes that the cell contains
    """
    def __init__(self):
        """
        """
        self.boxes = []
        self.content = ''
    
    def add_box(self, box: Box) -> None:
        self.boxes.append(box)

    def get_boxes(self) -> List[Box]:
        return self.boxes
    
    def set_content(self, content: str) -> None:
        self.content = content
    
    def get_content(self) -> str:
        return self.content

    def is_empty(self) -> bool:
        return len(self.boxes) == 0

    def _to_json(self):
        return {
            'boxes': [box._to_json() for box in self.boxes],
            'content': self.content
        }