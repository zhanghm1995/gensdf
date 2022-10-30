'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-28 10:32:30
Email: haimingzhang@link.cuhk.edu.cn
Description: The model for training PU task
'''
#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import math
from tqdm import tqdm

import os 
import os.path as osp
from pathlib import Path
import time 

from model import base_pl
from model.archs.encoders.conv_pointnet import ConvPointnet
from model.archs.decoders.deepsdf_arch import DeepSdfArch

from utils import mesh, evaluate
from pytorch3d.ops import sample_farthest_points
from pytorch3d.loss import chamfer_distance


class GenPU(base_pl.Model):
    def __init__(self, specs, dataloaders):
        super().__init__(specs)
        
        encoder_specs = self.specs["EncoderSpecs"]
        self.latent_size = encoder_specs["latent_size"]
        self.latent_hidden_dim = encoder_specs["hidden_dim"]
        self.unet_kwargs = encoder_specs["unet_kwargs"]
        self.plane_resolution = encoder_specs["plane_resolution"]

        decoder_specs = self.specs["DecoderSpecs"]
        self.decoder_hidden_dim = decoder_specs["hidden_dim"]
        self.skip_connection = decoder_specs["skip_connection"]
        self.geo_init = decoder_specs["geo_init"]

        lr_specs = self.specs["LearningRate"]
        self.lr_init = lr_specs["init"]
        self.lr_step = lr_specs["step_size"]
        self.lr_gamma = lr_specs["gamma"]

        self.alpha = self.specs["Alpha"]

        
        self.dataloaders = dataloaders

        self.build_model()


    def build_model(self):
        self.encoder = ConvPointnet(c_dim=self.latent_size, hidden_dim=self.latent_hidden_dim, 
                                        plane_resolution=self.plane_resolution,
                                        unet=(self.unet_kwargs is not None), unet_kwargs=self.unet_kwargs)
        
        self.decoder = DeepSdfArch(self.latent_size, self.decoder_hidden_dim, geo_init=self.geo_init, 
                                   skip_connection=self.skip_connection, out_dim=3)


    def configure_optimizers(self):
    
        optimizer = torch.optim.Adam(self.parameters(), self.lr_init)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, self.lr_step, self.lr_gamma)

        return [optimizer], [lr_scheduler]

 
    # context and queries from labeled, unlabeled data, respectively 
    def training_step(self, x, batch_idx):

        sparse_pc = x['point_cloud']
        query_points = x['sdf_xyz']
        gt_pc = x['gt_pc']

        # print(sparse_pc.dtype, query_points.dtype, gt_pc.dtype)
        # print(sparse_pc.shape, query_points.shape, gt_pc.shape)

        lab_shape_vecs = self.encoder(sparse_pc, query_points)
        #lab_enc_xyz = self.ff_enc(context_xyz, self.avals.to(self.device), self.bvals.to(self.device))
        lab_decoder_input = torch.cat([lab_shape_vecs, query_points], dim=-1)
        query_offsets = self.decoder(lab_decoder_input)
        
        # offset regulization loss
        offset_reg_loss = self.alpha * nn.L1Loss()(query_offsets, torch.zeros_like(query_offsets))
        pred_pts = query_points + query_offsets

        chamfer_dist_loss, final_pred_pts = self.chamfer_dist_loss(pred_pts, gt_pc)
        chamfer_dist_loss *= 100.0

        loss_dict =  {
                        "offset_reg_loss": offset_reg_loss,
                        "chamfer_dist_loss": chamfer_dist_loss,
                    }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)
        loss = chamfer_dist_loss + offset_reg_loss
        
        ## add visualization
        if batch_idx % 500 == 0:
            save_dir = osp.join(self.logger.log_dir, "vis", f"epoch_{self.current_epoch:04d}")
            os.makedirs(save_dir, exist_ok=True)

            sparse_pc_arr = sparse_pc.detach().cpu().numpy()
            pred_pts_arr = pred_pts.detach().cpu().numpy() # (B, N, 3)
            gt_pc_arr = gt_pc.detach().cpu().numpy()
            query_point_arr = query_points.detach().cpu().numpy()
            final_pred_pts_arr = final_pred_pts.detach().cpu().numpy()
            for i in range(pred_pts_arr.shape[0]):
                np.savetxt(osp.join(save_dir, f"{i:02d}_input.xyz"), sparse_pc_arr[i])
                np.savetxt(osp.join(save_dir, f"{i:02d}_query.xyz"), query_point_arr[i])
                np.savetxt(osp.join(save_dir, f"{i:02d}_pred.xyz"), pred_pts_arr[i])
                np.savetxt(osp.join(save_dir, f"{i:02d}_final_pred.xyz"), final_pred_pts_arr[i])
                np.savetxt(osp.join(save_dir, f"{i:02d}_gt.xyz"), gt_pc_arr[i])

        return loss

    def chamfer_dist_loss(self, pred_pt, gt_pc):

        if pred_pt.shape[1] > gt_pc.shape[1]:
            pred_pt, _ = sample_farthest_points(pred_pt, K=gt_pc.shape[1])
        loss, _ = chamfer_distance(pred_pt, gt_pc, norm=2)
        return loss, pred_pt

    def train_dataloader(self):
        if len(self.dataloaders)==1:
            return self.dataloaders[0]
        return self.dataloaders

    def forward(self, pc, query):
        shape_vecs = self.encoder(pc, query)
        decoder_input = torch.cat([shape_vecs, query], dim=-1)
        pred_sdf = self.decoder(decoder_input)

        return pred_sdf

    def reconstruct(self, model, test_data, eval_dir, testopt=True, sampled_points=15000):
        recon_samplesize_param = 256
        recon_batch = 1000000

        gt_pc = test_data['point_cloud'].float()
        #print("gt pc shape: ",gt_pc.shape)
        sampled_pc = gt_pc[:,torch.randperm(gt_pc.shape[1])[0:15000]]
        #print("sampled pc shape: ",sampled_pc.shape)

        if testopt:
            start_time = time.time()
            model = self.fast_opt(model, sampled_pc, num_iterations=800)

        model.eval() 
        

        with torch.no_grad():
            Path(eval_dir).mkdir(parents=True, exist_ok=True)
            mesh_filename = os.path.join(eval_dir, "reconstruct") #ply extension added in mesh.py
            evaluate_filename = os.path.join("/".join(eval_dir.split("/")[:-2]), "evaluate.csv")
            
            mesh_name = test_data["mesh_name"]

            levelset = 0.005 if testopt else 0.0
            mesh.create_mesh(model, sampled_pc, mesh_filename, recon_samplesize_param, recon_batch, level_set=levelset)
            try:
                evaluate.main(gt_pc, mesh_filename, evaluate_filename, mesh_name) # chamfer distance
            except Exception as e:
                print(e)

    def fast_opt(self, model, full_pc, num_iterations=800):

        num_iterations = num_iterations
        xyz_full, gt_pt_full = self.fast_preprocess(full_pc)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print("performing refinement on input point cloud...")
        #print("shapes: ", full_pc.shape, xyz_full.shape)
        for e in tqdm(range(num_iterations)):
            samp_idx = torch.randperm(xyz_full.shape[1])[0:5000]
            xyz = xyz_full[ :,samp_idx ].cuda()
            gt_pt = gt_pt_full[ :,samp_idx ].cuda()
            pc = full_pc[:,torch.randperm(full_pc.shape[1])[0:5000]].cuda()

            shape_vecs = model.encoder(pc, xyz)
            decoder_input = torch.cat([shape_vecs, xyz], dim=-1)
            pred_sdf = model.decoder(decoder_input).unsqueeze(-1)

            pc_vecs = model.encoder(pc, pc)
            pc_pred = model.decoder(torch.cat([pc_vecs, pc], dim=-1))

            pred_pt, gt_pt = model.get_unlab_offset(xyz, gt_pt, pred_sdf)

            # loss of pt offset and loss of L1
            unlabeled_loss = nn.MSELoss()(pred_pt, gt_pt)
            # using pc to supervise query as well
            pc_l1 = nn.L1Loss()(pc_pred, torch.zeros_like(pc_pred))

            loss = unlabeled_loss + 0.01*pc_l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return model


    def fast_preprocess(self, pc):
        pc = pc.squeeze()
        pc_size = pc.shape[0]
        query_per_point=20

        def gen_grid(start, end, num):
            x = np.linspace(start,end,num=num)
            y = np.linspace(start,end,num=num)
            z = np.linspace(start,end,num=num)
            g = np.meshgrid(x,y,z)
            positions = np.vstack(map(np.ravel, g))
            return positions.swapaxes(0,1)

        dot5 = gen_grid(-0.5,0.5, 70) 
        dot10 = gen_grid(-1.0, 1.0, 50)
        grid = np.concatenate((dot5,dot10))
        grid = torch.from_numpy(grid).float()
        grid = grid[ torch.randperm(grid.shape[0])[0:30000] ]

        total_size = pc_size*query_per_point + grid.shape[0]

        xyz = torch.empty(size=(total_size,3))
        gt_pt = torch.empty(size=(total_size,3))

        # sample xyz
        dists = torch.cdist(pc, pc)
        std, _ = torch.topk(dists, 50, dim=-1, largest=False)
        std = std[:,-1].unsqueeze(-1)

        count = 0
        for idx, p in enumerate(pc):
            # query locations from p
            q_loc = torch.normal(mean=0.0, std=std[idx].item(),
                                 size=(query_per_point, 3))

            # query locations in space
            q = p + q_loc
            xyz[count:count+query_per_point] = q
            count += query_per_point

    
        xyz[pc_size*query_per_point:] = grid 

        # nearest neighbor
        dists = torch.cdist(xyz, pc)
        _, min_idx = torch.min(dists, dim=-1)  
        gt_pt = pc[min_idx]
        return xyz.unsqueeze(0), gt_pt.unsqueeze(0)



    

