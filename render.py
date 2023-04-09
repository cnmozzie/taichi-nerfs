from opt import get_opts
from modules.networks import TaichiNGP
from modules.rendering import render
from modules.utils import load_ckpt
from datasets.ray_utils import get_rays, get_ray_directions
import taichi as ti
import torch
import numpy as np
from kornia.utils.grid import create_meshgrid3d
from einops import rearrange
import imageio
import time

def taichi_init(args):
    taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
    if args.half2_opt:
        taichi_init_args["half2_vectorization"] = True

    ti.init(**taichi_init_args)

def get_directions(fx, fy, w, h):
    K = np.float32([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])
    K = torch.FloatTensor(K)
    directions = get_ray_directions(h, w, K)
    return directions.to(torch.device('cuda'))

def render_image(model, directions, poses, w, h, kwargs):
    rays_o, rays_d = get_rays(directions, poses)
    results = render(model, rays_o, rays_d, **kwargs)
    rgb_pred = rearrange(results['rgb'], '(h w) c -> h w c', h=h)
    return rgb_pred

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

kwargs = {
    'test_time': True,
    'random_bg': hparams.random_bg
}
if hparams.scale > 0.5:
    kwargs['exp_step_factor'] = 1 / 256

fx = fy = 1111.1110311937682
w = h = 800
directions = get_directions(fx, fy, w, h)

t1 = time.time()
poses = torch.tensor([[-1.0000, -0.0000, -0.0000,  0.0117],
        [ 0.0000,  0.7341, -0.6790,  1.0360],
        [ 0.0000, -0.6790, -0.7341,  1.0072]])

poses1 = torch.tensor([[-0.6566542387008667, 0.2367822825908661, -0.7160581946372986, 2.8865230083465576],
                       [0.7541918158531189, 0.20615987479686737, -0.623452365398407, 2.513216972351074],
                       [-1.4901162970204496e-08, -0.9494377970695496, -0.31395500898361206, 1.26559317111969]])
poses = poses.to(torch.device('cuda'))
poses1 = poses1.to(torch.device('cuda'))

rgb_pred = render_image(model, directions, poses, w, h, kwargs)
rgb_pred = render_image(model, directions, poses1, w, h, kwargs)

t3 = time.time()

print(t3-t1)

imageio.imsave(f'rgb_test.png', rgb_pred.cpu().numpy())
