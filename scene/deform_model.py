import os
import torch
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from scene.videoartgs import VideoArtGS


class DeformModel:
    def __init__(self, args):
        self.deform = VideoArtGS(args).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    @property
    def reg_loss(self):
        return self.deform.reg_loss
    
    def init_from_joint_info(self, joint_infos, init_joint_info=True, init_center=True):
        self.deform.init_from_joint_info(joint_infos, init_joint_info, init_center)

    def step(self, gaussians, is_training=True):
        return self.deform(gaussians, is_training=is_training)

    def train_setting(self, training_args):
        l = []
        for group in self.deform.trainable_parameters():
            lr = training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale
            l.append({
                'params': group['params'],
                'lr': lr,
                "name": group['name']
            })
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deform_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale, 
                                                  lr_final=training_args.position_lr_final * training_args.deform_lr_scale, 
                                                  lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "deform/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))
            with open(os.path.join(out_weights_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
            torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))
        
    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        if os.path.exists(weights_path):
            self.deform.load_state_dict(torch.load(weights_path))
            return True
        else:
            return False

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.deform_scheduler(iteration)
        
    def update(self, iteration):
        self.deform.update(iteration)




