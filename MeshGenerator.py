import numpy as np
import open3d as o3d

def formatCloud(pointCloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointCloud[:, :3].astype(float))
    pcd.colors = o3d.utility.Vector3dVector(pointCloud[:, 4:].astype(float))

    # not given normals so we estimate
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return pcd

def gen3d(pcd, method='pointcloud'):
    if method == 'mesh':
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        o3d.visualization.draw_geometries([mesh])
    else:
        o3d.visualization.draw_geometries([pcd])

