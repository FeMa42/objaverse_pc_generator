import os
import signal

import open3d as o3d

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

import pandas as pd
# argparse is a module that allows you to pass arguments to your script
import argparse


def point_cloud_generation_from_mesh(mesh_path, pc_full_path=None):
    # Load the mesh from a .glb file
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # Sample points on the mesh surface
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=10000)

    if pc_full_path is not None:
        try:
            o3d.io.write_point_cloud(pc_full_path, point_cloud)
        except:
            return None

    # return point_cloud

def generate_point_cloud(uid, mesh_path, base_dir):
    pc_full_path = os.path.join(base_dir, f"{uid}.ply")
    point_cloud_generation_from_mesh(mesh_path, pc_full_path)


def generate_point_clouds_from_class(class_name, loaded_objects, max_workers=8):
    base_dir = class_name + "_point_clouds"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # # go through all plane_objects and generate point clouds
    # skipped = 0
    # for uid, mesh_path in tqdm(loaded_objects.items()):
    #     pc_full_path = os.path.join(base_dir, f"{uid}.ply")

    #     returned_pc = point_cloud_generation_from_mesh(mesh_path, pc_full_path)

    #     if returned_pc is None:
    #         skipped += 1

    # print(f"Skipped {skipped} objects.")

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
