import torch


def index(feat, uv):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [0, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)    # [B, N, 2]

    (B, N, _) = uv.shape
    C = feat.shape[1]

    if uv.shape[-1] == 3:
        # uv = uv[:,:,[2,1,0]]
        # uv = uv * torch.tensor([1.0,-1.0,1.0]).type_as(uv)[None,None,...]
        uv = uv.unsqueeze(2).unsqueeze(3)    # [B, N, 1, 1, 3]
    else:
        uv = uv.unsqueeze(2)    # [B, N, 1, 2]

    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)    # [B, C, N, 1]
    return samples.view(B, C, N)    # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)    # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts

def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:, :2, :2]
        shift = transforms[:, :2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz

if __name__ == '__main__':
    import cv2
    import numpy as np
    img = cv2.imread('/home/xiaojin/projects/refinement/pifuhd/lib/geo2.png')
    inp = torch.from_numpy(np.moveaxis(img, -1, 0)).unsqueeze(0).float()
    out_h = 1000
    out_w = 1000
    new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
    new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
    grid = torch.cat((new_w.unsqueeze(2), -new_h.unsqueeze(2)), dim=2)
    grid = grid.reshape(-1, 2).T.unsqueeze(0)
    print(inp.shape)
    print(grid.shape)
    out = index(inp, grid).squeeze(0).T.numpy()
    out = out.reshape(out_w, out_h, -1)
    print(out.shape)
    # out = np.moveaxis(out, 0, -1)
    # cv2.imwrite("in.png", img)
    cv2.imwrite("geo.png", out)
    print(out.shape)