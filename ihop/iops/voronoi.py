""" Methods related to Convex Hull tesselation 

References:
  https://books.google.de/books?hl=de&lr=&id=cic8DQAAQBAJ&oi=fnd&pg=PR5&dq=voronoi+diagrams+and+delaunay+triangulations&ots=O7pZlG8EDz&sig=z5d1wRFEdsVRbf-9o2agA-pYwvw#v=onepage&q=voronoi%20diagrams%20and%20delaunay%20triangulations&f=false

"""

import numpy as np

from scipy.spatial import Delaunay


def build_delaunay(points:np.ndarray):
    """ Build a Delaunay triangulation from a set of points

    Args:
        points (np.ndarray): points to triangulate

    Returns:
        scipy.spatial.Delaunay: triangulation
    """

    return Delaunay(points)