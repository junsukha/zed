import numpy as np
import open3d as o3d

def key_callback(vis):
    print('key')

points = np.random.rand(100000, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(point_cloud)
vis.register_key_callback(ord("a"), key_callback)

vis.run()
vis.destroy_window()