from SPACE.engine.utils import get_config
from SPACE.utils import Checkpointer
from SPACE.model import get_model
import torch
import numpy as np

def load_space_model(checkpoint_path, check_name, device):
    cfg = get_config()
    model = get_model(cfg)
    model = model.to(device)
    checkpointer = Checkpointer(checkpoint_path, max_num=cfg.train.max_ckpt)
    use_cpu = 'cpu' in device
    checkpointer.load(check_name, model, None, None, use_cpu=use_cpu)
    return model


'''if __name__ == '__main__':
    device = 'cuda:0'
    model = load_space_model(checkpoint_path='../data/FetchGenerativeEnv-v1/',
                         check_name='../data/FetchGenerativeEnv-v1/model_000030001.pth', device=device)
    model.eval()

    from SPACE.engine.utils import bbox_in_one
    from attrdict import AttrDict
    from torchvision.utils import make_grid
    import imageio
    from torch.utils.data import Subset, DataLoader
    import time
    data_set = np.load('../data/FetchGenerativeEnv-v1/all_set_val.npy')
    data_size = len(data_set)
    idx_set = np.arange(data_size)
    np.random.shuffle(idx_set)
    idx_set = np.split(idx_set, len(idx_set) / 4)
    data = data_set[idx_set[0]]
    data = torch.from_numpy(data).float().to(device)
    data /= 255
    data = data.permute([0, 3, 1, 2])
    loss, log = model(data, 100000000)
    for key, value in log.items():
        if isinstance(value, torch.Tensor):
            log[key] = value.detach().cpu()
    log = AttrDict(log)
    # (B, 3, H, W)
    fg_box = bbox_in_one(
        log.fg, log.z_pres, log.z_scale, log.z_shift
    )
    # (B, 1, 3, H, W)
    imgs = log.imgs[:, None]
    fg = log.fg[:, None]
    recon = log.y[:, None]
    fg_box = fg_box[:, None]
    bg = log.bg[:, None]
    # (B, K, 3, H, W)
    comps = log.comps
    # (B, K, 3, H, W)
    masks = log.masks.expand_as(comps)
    masked_comps = comps * masks
    alpha_map = log.alpha_map[:, None].expand_as(imgs)
    grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
    nrow = grid.size(1)
    B, N, _, H, W = grid.size()
    grid = grid.view(B * N, 3, H, W)

    # (3, H, W)
    grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)

    # (H, W, 3)
    image = torch.clamp(grid_image, 0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    imageio.imwrite('table.png', image)'''
