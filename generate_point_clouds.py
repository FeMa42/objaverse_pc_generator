import os
import signal

# import open3d as o3d
import trimesh

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

import pandas as pd
# argparse is a module that allows you to pass arguments to your script
import argparse


def point_cloud_generation_from_mesh(mesh_path, pc_full_path=None):
    # Load the mesh
    mesh = trimesh.load(mesh_path, force='mesh')

    # Check if the mesh is loaded correctly
    if not mesh.is_empty:
        print("Mesh loaded successfully!")
    else:
        print("Failed to load the mesh.")

    # Translate the mesh to the origin
    mesh.apply_translation(-mesh.centroid)

    # Scale the mesh to fit within a cube from -1 to 1
    mesh.apply_scale(1 / (mesh.scale*0.5))

    # Sample points on the mesh surface
    n_points = 10000
    points, faces = mesh.sample(n_points, return_index=True)
    normals = mesh.face_normals[faces]

    # concatenate the points and normals
    point_cloud = np.concatenate([points, normals], axis=1)

    # save the point cloud to a npy file
    np.save(pc_full_path, point_cloud)

    # return point_cloud
    return point_cloud

def generate_point_cloud(uid, mesh_path, base_dir):
    pc_full_path = os.path.join(base_dir, f"{uid}.npy")
    print(f"Generating point cloud for {uid} from {mesh_path}")
    pct = point_cloud_generation_from_mesh(mesh_path, pc_full_path)
    return pct


def generate_point_clouds_from_class(class_name, loaded_objects, max_workers=8):
    base_dir = class_name + "_point_clouds"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_point_cloud, uid, mesh_path, base_dir): uid
                   for uid, mesh_path in loaded_objects.items()}

        for future in tqdm(as_completed(futures), total=len(futures)):
            uid = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {uid}: {e}")

    return base_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters

    # path to the test ply
    parser.add_argument('--class_name', type=str,
                        default='bicycle')

    config = parser.parse_args()

    class_name = config.class_name

    loaded_objects = pd.read_csv(
        class_name + "_objects.csv", index_col=0).to_dict()["0"]
    print(
        f"Generating point clouds for {len(loaded_objects)} {class_name} objects.")
    generate_point_clouds_from_class(class_name, loaded_objects, max_workers=32)
