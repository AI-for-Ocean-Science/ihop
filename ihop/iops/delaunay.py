""" Methods related to Convex Hull tesselation 

References:
  https://books.google.de/books?hl=de&lr=&id=cic8DQAAQBAJ&oi=fnd&pg=PR5&dq=voronoi+diagrams+and+delaunay+triangulations&ots=O7pZlG8EDz&sig=z5d1wRFEdsVRbf-9o2agA-pYwvw#v=onepage&q=voronoi%20diagrams%20and%20delaunay%20triangulations&f=false

X and Bard:
  https://g.co/bard/share/9783a3d76efe

"""

import numpy as np

from scipy.spatial import Delaunay


def build_delaunay(points:np.ndarray):
    """ Build a Delaunay triangulation from a set of points

    Args:
        points (np.ndarray): points to triangulate (npoints, ndim)

    Returns:
        scipy.spatial.Delaunay: triangulation
    """

    return Delaunay(points)

def in_the_hull(points:np.ndarray, hull:Delaunay):
    """ Find the points in the hull

    Args:
        points (np.ndarray): points to test
        hull (Delaunay): hull to test

    Returns:
        np.ndarray: points in the hull
    """

    return hull.find_simplex(points)>=0