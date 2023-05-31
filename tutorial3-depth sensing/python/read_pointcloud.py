import open3d as o3d
 
pcd = o3d.io.read_point_cloud("Pointcloud.pcd")
pcd1 = o3d.io.read_point_cloud("Pointcloud1.pcd")
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
# vis.add_geometry(pcd1)
vis.run()