import numpy as np
import copy

class Box:
    """ Represents the bounding box of the polygon passed """
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
    
    def get_X(self):
        return self.X.copy()
    
    def get_Y(self):
        return self.Y.copy()

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


    def __str__(self):
        return str(self.vertices.tolist())

    def __eq__(self, other):
        return np.allclose(self.get_X_range(), other.get_X_range()) and np.allclose(self.get_Y_range(), other.get_Y_range())

    def __hash__(self):
        return hash(str(self.vertices))