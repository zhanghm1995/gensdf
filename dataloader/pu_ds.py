'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-28 21:24:46
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset for Point Cloud Upsampling task
'''
#!/usr/bin/env python3

import numpy as np
import time 
from tqdm import tqdm
import logging
import os
import random
import torch
import torch.utils.data

import pandas as pd 
import csv


class PUDS(torch.utils.data.Dataset): 

    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 

        query_per_point = 20, 
        pc_size = 5000,
        samples_per_batch = 16000,

        samples_per_mesh=100000, # 5000*20 from pc

        load_files=False,
        save_files=False,
        save_dir = "preprocessed"
    ):

        self.samples_per_mesh = samples_per_mesh
        self.samples_per_batch = samples_per_batch
        self.pc_size = pc_size
        self.query_per_point = query_per_point
        self.num_iters = int(samples_per_mesh / samples_per_batch)

        self.unlab_files = self.get_instance_filenames(data_source, split_file)
        self.unlab_files = self.unlab_files[6370:]

        if load_files:
            print("Loading from saved data...")
            prep_data = torch.load("{}/prep_data.pt".format(save_dir))
            self.point_clouds = prep_data["point_cloud"]
            self.sdf_xyz = prep_data["sdf_xyz"]
            self.gt_pc = prep_data["gt_pc"]
        else:
            self.sdf_xyz      = torch.empty(size=(len(self.unlab_files), samples_per_mesh, 3))
            self.query_projection = torch.empty(size=(len(self.unlab_files), samples_per_mesh, 3))
            self.point_clouds = torch.empty(size=(len(self.unlab_files), pc_size, 3))
            self.gt_pc        = torch.empty(size=(len(self.unlab_files), 20000, 3))

            self.preprocess_data()

            if save_files:
                print("Saving files...")
                os.makedirs(save_dir, exist_ok=True)
                torch.save( {
                            "point_cloud":self.point_clouds,
                            "sdf_xyz":self.sdf_xyz,
                            "gt_pc":self.gt_pc,
                            "query_projection":self.query_projection,
                            },
                            "{}/prep_pu_data.pt".format(save_dir))
            
        l = self.sdf_xyz.shape[0]
        self.sdf_xyz = self.sdf_xyz.reshape(l, -1, self.samples_per_mesh, 3) # TODO: use self.samples_per_batch originally
        self.gt_pc = self.gt_pc.reshape(l, -1, 20000, 3)

        # # point clouds should just be repeated 
        self.point_clouds = self.point_clouds.unsqueeze(1).repeat(1, self.gt_pc.shape[1], 1, 1)
        print("pc, xyz, pt shapes: ", self.point_clouds.shape, self.sdf_xyz.shape, self.gt_pc.shape)
           

    def __getitem__(self, idx):
        return {
                "sdf_xyz":self.sdf_xyz[idx].float().squeeze(),
                "gt_pc":self.gt_pc[idx].float().squeeze(),
                "point_cloud":self.point_clouds[idx].float().squeeze(),
                "indices": idx
        }


    def preprocess_data(self):
        with tqdm(self.unlab_files) as pbar:
            for i, f in enumerate(pbar):
                pbar.set_description("Files processed: {}/{}: {}".format(i, len(self.unlab_files), f))
                gt_pc = torch.from_numpy(np.loadtxt(f).astype(np.float32)) # (20000, 3)
                gt_pc = self.normalize_pointcloud(gt_pc)

                pc = self.sample_pointcloud(gt_pc) # (5000, 3)
                query_points = self.sample_query(pc) # (100000, 3)
                # find the nearset projection points of query points in the gt point cloud
                nearest_neighbors, _ = self.find_nearest_query_neighbor(gt_pc, query_points)

                self.point_clouds[i] = pc # input sparse point cloud
                self.sdf_xyz[i] = query_points
                self.query_projection[i] = nearest_neighbors
                self.gt_pc[i] = gt_pc

    def normalize_pointcloud(self, f):
        f -= torch.mean(f, dim=0)
        bbox_length = torch.sqrt(torch.sum((torch.max(f, axis=0)[0] - torch.min(f, axis=0)[0])**2))
        f /= bbox_length
        return f

    def sample_pointcloud(self, f):
        pc_idx = torch.randperm(f.shape[0])[0:self.pc_size]

        return f[pc_idx].float()

    def get_instance_filenames(self, data_source, split, gt_filename="gt_pc.xyz"):
        xyzfiles = []
        for dataset in split: # e.g. "acronym" "shapenet"
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:
                    instance_filename = os.path.join(data_source, dataset, class_name, instance_name, gt_filename)
                    
                    if not os.path.isfile(instance_filename):
                        logging.warning("Requested non-existent file '{}'".format(instance_filename))
                        continue

                    xyzfiles.append(instance_filename)
        return xyzfiles

    # find the 50th nearest neighbor for each point in pc 
    # this will be the std for the gaussian for generating query 
    def sample_query(self, pc):

        dists = torch.cdist(pc, pc)

        std, _ = torch.topk(dists, 50, dim=-1, largest=False)

        std = std[:,-1].unsqueeze(-1)

        query_points = torch.empty(size=(pc.shape[0]*self.query_per_point, 3))
        count = 0

        for idx, p in enumerate(pc):

            # query locations from p
            q_loc = torch.normal(mean=0.0, std=std[idx].item(),
                                 size=(self.query_per_point, 3))

            # query locations in space
            q = p + q_loc

            query_points[count:count+self.query_per_point] = q

            count += self.query_per_point

        return query_points

    # the closest point in the pc for all query points 
    def find_nearest_query_neighbor(self, pc, query_points):

        dists = torch.cdist(query_points, pc)
        min_dist, min_idx = torch.min(dists, dim=-1)  
        nearest_neighbors = pc[min_idx]

        return nearest_neighbors, min_dist.unsqueeze(-1)



    def __len__(self):
        return len(self.unlab_files)

class Sampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shape_num):
        super().__init__(data_source)
        self.data_source = data_source
        self.shape_num = shape_num

    def __iter__(self):
        rt = np.random.choice(self.data_source.gt_pt.shape[1], self.data_source.gt_pt.shape[1], replace=False)
        rt = torch.from_numpy(rt)
        iter_order = [(i, rt[j]) for i in range(self.shape_num) for j in range(self.data_source.gt_pt.shape[1])]
        return iter(iter_order)

    def __len__(self):
        return self.data_source.gt_pt.shape[0] * self.data_source.gt_pt.shape[1]


