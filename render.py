from opt import get_opts
from modules.networks import TaichiNGP
from modules.rendering import render
from modules.utils import load_ckpt
from datasets.ray_utils import get_rays
import taichi as ti
import torch
from kornia.utils.grid import create_meshgrid3d
from datasets import dataset_dict
from einops import rearrange
import imageio

def taichi_init(args):
    taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
    if args.half2_opt:
        taichi_init_args["half2_vectorization"] = True

    ti.init(**taichi_init_args)

# 参考datasets/nsvf.py改写这个函数
# 使其能根据相机内参推导出direction数组
def get_directions():
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {
        'root_dir': hparams.root_dir,
        'downsample': hparams.downsample
    }
    test_dataset = dataset(split='test', **kwargs)
    return test_dataset.directions.to(torch.device('cuda'))

hparams = get_opts()
taichi_init(hparams)


rgb_act = 'Sigmoid'
model = TaichiNGP(hparams, scale=hparams.scale, rgb_act=rgb_act)

G = model.grid_size
model.register_buffer('density_grid', torch.zeros(model.cascades, G**3))
model.register_buffer(
    'grid_coords',
    create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
load_ckpt(model, hparams.ckpt_path)
model.to(torch.device('cuda'))


directions = get_directions()
poses = torch.tensor([[-1.0000, -0.0000, -0.0000,  0.0117],
        [ 0.0000,  0.7341, -0.6790,  1.0360],
        [ 0.0000, -0.6790, -0.7341,  1.0072]])
poses = poses.to(torch.device('cuda'))

rays_o, rays_d = get_rays(directions, poses)

kwargs = {
    'test_time': True,
    'random_bg': hparams.random_bg
}
if hparams.scale > 0.5:
    kwargs['exp_step_factor'] = 1 / 256
results = render(model, rays_o, rays_d, **kwargs)

w = 800
h = 800
rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
imageio.imsave(f'rgb_test.png', rgb_pred)
