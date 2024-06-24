import sys
import time
sys.path.append(".")
import os
import numpy as np
import torch
from dataset.dataset_3dpw import batch_denormalization, SoMoFDataset_3dpw_test
from model.model import JRTransformer

from torch.utils.data import DataLoader, SequentialSampler, DataLoader
from utils.config_3dpw import *
from utils.metrics import batch_MPJPE, batch_VIM
from utils.util import get_adj, get_connect

from datetime import datetime


from dataset.dataset_skelda import SkeldaDataset
SoMoFDataset_3dpw_test = SkeldaDataset

class Tester:
    def __init__(self, args):
        # Set cuda device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(0)
        else:
            self.device = torch.device("cpu")
        print('Using device:', self.device)
        self.cuda_devices = args.device

        # Parameters
        self.batch_size = 1

        # Defining models
        self.model = JRTransformer(N=args.N, J=args.J, in_joint_size=args.in_joint_size, in_relation_size=args.in_relation_size, 
                                   feat_size=args.d_k, out_joint_size=args.out_joint_size, out_relation_size=args.out_relation_size,
                                   num_heads=args.num_heads, depth=args.depth).to(self.device)
        
        self.rc = args.rc
        dset_test = SoMoFDataset_3dpw_test(dset_path=somof_3dpw_test_data, seq_len=args.input_length+args.output_length, N=args.N, J=args.J, split_name="test")
        sampler_test = SequentialSampler(dset_test)
        self.test_loader = DataLoader(dset_test, sampler=sampler_test, batch_size=self.batch_size, num_workers=2, drop_last=False, pin_memory=True)
        
        edges = [(0, 1), (1, 8), (8, 7), (7, 0),
			 (0, 2), (2, 4),
			 (1, 3), (3, 5),
			 (7, 9), (9, 11),
			 (8, 10), (10, 12),
			 (6, 7), (6, 8)]
        self.adj = get_adj(args.N, args.J, edges)
        self.adj = self.adj.unsqueeze(0).unsqueeze(-1)
        self.conn = get_connect(args.N, args.J)
        self.conn = self.conn.unsqueeze(0).unsqueeze(-1)
        self.Tt = args.input_length + args.output_length
        self.Ti = args.input_length
        self.To = args.output_length
        self.J = args.J

        self.path = args.model_path
        
    def test(self):
        path = self.path
        checkpoint = torch.load(path)  
        self.model.load_state_dict(checkpoint['net']) 
        self.model.eval()

        all_mpjpe = np.zeros(self.To)
        all_vim = np.zeros(self.To)
        count = 0
        stime = time.time()

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                input_total_original, para = data
                input_total_original = input_total_original.float().cuda()
                input_total = input_total_original.clone()

                batch_size = input_total.shape[0]
                input_total[..., [1, 2]] = input_total[..., [2, 1]]
                input_total[..., [4, 5]] = input_total[..., [5, 4]]

                if self.rc:
                    camera_vel = input_total[:, 1:self.Tt, :, :, 3:].mean(dim=(1, 2, 3)) # B, 3
                    input_total[..., 3:] -= camera_vel[:, None, None, None]
                    input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:].cumsum(dim=1)

                input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, self.Tt, 6)
				# B, NxJ, T, 6

                input_joint = input_total[:,:, :self.Ti]
				
                pos = input_total[:,:,:self.Ti,:3]
                pos_i = pos.unsqueeze(-3)
                pos_j = pos.unsqueeze(-4)
                pos_rel = pos_i - pos_j
                dis = torch.pow(pos_rel, 2).sum(-1)
                dis = torch.sqrt(dis)
                exp_dis = torch.exp(-dis)
                input_relation = torch.cat((exp_dis, self.adj.repeat(batch_size, 1, 1, 1), self.conn.repeat(batch_size, 1, 1, 1)), dim=-1)

                pred_vel = self.model.predict(input_joint, input_relation)
                pred_vel = pred_vel[:, :, self.Ti:]
                pred_vel = pred_vel.permute(0, 2, 1, 3)

                if self.rc:
                    pred_vel = pred_vel + camera_vel[:, None, None]

				# B, T, NxJ, 3
                pred_vel[..., [1, 2]] = pred_vel[..., [2, 1]]
				# Cumsum velocity to position with initial pose.
                motion_gt = input_total_original[...,:3].view(batch_size, self.Tt, -1, 3)
                motion_pred = (pred_vel.cumsum(dim=1) + motion_gt[:, self.Ti-1:self.Ti])
				
				# Apply denormalization.
                motion_pred = batch_denormalization(motion_pred.cpu(), para).numpy()               
                motion_gt = batch_denormalization(motion_gt.cpu(), para).numpy()

                # motion_inp = motion_gt[:, :self.Ti, :self.J, :]
                motion_gt = motion_gt[:, self.Ti:, :self.J, :]
                motion_pred = motion_pred[:, :, :self.J, :]

                trange = list(range(self.To))
                metric_MPJPE = batch_MPJPE(motion_gt, motion_pred, trange)
                all_mpjpe += metric_MPJPE

                metric_VIM = batch_VIM(motion_gt, motion_pred, trange)
                all_vim += metric_VIM

                # if count % 1000 == 0:
                #     print(motion_gt.shape)
                #     print(motion_gt[0, -1])
                #     print(motion_pred[0, -1])
                #     print(metric_MPJPE)
                #     from dataset import vis_skelda
                #     motion_inp = None
                #     vis_skelda.visualize(motion_inp, motion_gt, motion_pred)
                
                count += batch_size

            ttime = time.time()-stime
            print("Total time:", ttime, "FPS:", count/ttime)

            all_mpjpe *= 1000
            all_vim *= 1000
            all_mpjpe /= count
            all_vim /= count
            print('Test MPJPE:\t avg: {:.2f}'.format(all_mpjpe.mean()))
            print(all_mpjpe.astype(float).round(2))
            print('Test VIM:\t avg: {:.2f}'.format(all_vim.mean())) 
        return all_vim.mean()

if __name__=='__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    tester = Tester(args)
    tester.test()