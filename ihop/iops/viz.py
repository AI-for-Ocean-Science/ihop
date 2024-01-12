""" Scripts to visualize the Vernooi space of IOP decompositions """

import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from bokeh.plotting import figure, show
from bokeh.models import CustomJS, ColumnDataSource

def plot_voronoi(points):
    vor = Voronoi(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], cmap='viridis', linewidth=0.2)
    plt.show()


def surface_plot(xs, ys, zs):

    # Define data source for x, y, and z coordinates
    source = ColumnDataSource(
        data=dict(x=xs, y=ys, z=zs)
        #data=dict(x=[0.5, 1, 0.5], y=[0.5, 0, 0.5], z=[1, 1, 0.5])
    )

    # Define custom JavaScript code for the surface
    js_code = """
    const vertices = new Float32Array(data.source.data['x'].length * 3);
    const faces = new Uint32Array(data.source.data['x'].length * 6);

    // Generate vertex data for a sphere
    for (let i = 0; i < data.source.data['x'].length; i++) {
    const theta = i * Math.PI * 2 / data.source.data['x'].length;
    const phi = Math.PI * 0.5;
    vertices[i * 3] = Math.cos(theta) * Math.sin(phi);
    vertices[i * 3 + 1] = Math.sin(theta) * Math.sin(phi);
    vertices[i * 3 + 2] = Math.cos(phi);
    }

    // Generate triangle faces for the sphere
    for (let i = 0; i < data.source.data['x'].length - 1; i++) {
    faces[i * 6] = i;
    faces[i * 6 + 1] = i + 1;
    faces[i * 6 + 2] = i + data.source.data['x'].length;
    faces[i * 6 + 3] = i + data.source.data['x'].length;
    faces[i * 6 + 4] = i + 1;
    faces[i * 6 + 5] = i + data.source.data['x'].length + 1;
    }

    const mesh = new THREE.Mesh(
    new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(vertices, 3)),
    new THREE.MeshBasicMaterial({color: 0xff0000})
    );

    scene.add(mesh);
    """

    # Create a Bokeh figure and add the surface renderer
    p = figure(width=500, height=250)
    p.x_range = (0, 1.5)
    p.y_range = (0, 1.5)
    p.z_range = (0, 1.5)

    # Create a customJS callback with the JavaScript code
    surface = CustomJS(args=dict(data=source), code=js_code)
    p.renderer.add_callback(surface)

    # Show the plot
    show(p)


# Command line
if __name__ == '__main__':

    points = np.random.rand(30, 3)  # 30 points in 3D

    # Matplotlib
    #plot_voronoi(points)

    surface_plot(points[:,0], points[:,1], points[:,2])