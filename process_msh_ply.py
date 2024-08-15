import os, argparse, time

import open3d as o3d

import time
import cv2
from PIL import Image
import numpy as np


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, patch_size):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    # get the first 6 FPS anchor points
    #npoint = int(N/patch_size) + 1
    npoint = 6
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def knn_patch(pcd_name, patch_size = 2048):
    pcd = o3d.io.read_point_cloud(pcd_name)
    # nomalize pc and set up kdtree
    points = pc_normalize(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    fps_point = farthest_point_sample(points,patch_size)
    
    
    point_size = fps_point.shape[0]


    patch_list = []

    for i in  range(point_size):
        [_,idx,dis] = kdtree.search_knn_vector_3d(fps_point[i],patch_size)
        #print(pc_normalize(np.asarray(point)[idx[1:], :]))
        patch_list.append(np.asarray(points)[idx[:], :]) 
    
    #visualize(all_point(np.array(patch_list)))
    #visualize(point)
    return np.array(patch_list)


def background_crop(img):
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print(gray_img.shape)
    col = np.mean(gray_img,axis=0)
    row = np.mean(gray_img,axis=1)
    for i in range(len(col)):
        if col[i] != 255:
            col_a = i
            break
    for i in range(len(col)):
        if col[-i] != 255:
            col_b = len(col)-i
            break  
    for i in range(len(row)):
        if row[i] != 255:
            row_a = i
            break
    for i in range(len(row)):
        if row[-i] != 255:
            row_b = len(row)-i
            break       
    img = img[row_a:row_b,col_a:col_b,:]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def ply2projections(objname):
    start = time.time()
    print(objname)
    pcd = o3d.io.read_point_cloud(objname)
    vis = o3d.visualization.Visualizer()


    vis.create_window(visible=False,width=1080,height=1920)
    vis.add_geometry(pcd)
    ctrl = vis.get_view_control()
    # set the view control so that whole object is in the view
    # ctrl.set_front([0,0,1])
    # ctrl.set_up([0,1,0])
    # ctrl.set_lookat([0,0,0])

    interval = 5.82 # interval for 1 degree
    start = time.time()
    imgs = []
    # begin rotation rotate the camera on the pathway of (x^2 + y^2 = r^2, z = 0) and (y^2 + z^2 = r^2, x = 0) 
    rotate_para = [[0,0],[90*interval,0],[90*interval,0],[90*interval,0]]
    for i in range(4):
        ctrl.rotate(rotate_para[i][0],rotate_para[i][1])
        ctrl.set_zoom(2.0)
        vis.poll_events()
        vis.update_renderer()    
        img = vis.capture_screen_float_buffer(True)
        img = Image.fromarray((np.asarray(img)* 255).astype(np.uint8))
        # crop the main object out of the white background
        img = background_crop(img)
        imgs.append(img)

    end = time.time()
    # print("time consuming: ",end-start)
    vis.destroy_window()
    del ctrl
    del vis
    return imgs


def main(config):
    #get the projections and patches
    print('Begin loading the ply file.')
    print(config.objname)
    imgs = ply2projections(config.objname)
    print('Projections generated.')
    patches = knn_patch(config.objname)
    print('Patches generated.')

    # save imgs and patches as numpy files for further analysis
    patches_filename = os.path.join(os.path.dirname(config.objname), 'patches.npy')
    np.save(patches_filename, patches)
    for i, img in enumerate(imgs):
        image_file_name = os.path.join(os.path.dirname(config.objname), 'image'+str(i)+'.npy')
        np.save(image_file_name, img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters

    parser.add_argument('--objname', type=str, default='/bag/bag_level_7.ply') # path to the test ply
    parser.add_argument('--ckpt_path', type=str, default='WPC.pth') # path to the pretrained weights

    config = parser.parse_args()

    print("start testing...")

    main(config)
