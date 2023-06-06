import collections
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import io



Batch_Description = collections.namedtuple(
    "Batch_Description",
    ("y_values", "x_values",
     "target_y", "target_x",
     "context_y", "context_x")
    )


def to_cuda(batch):
    return Batch_Description(
            x_values=batch.x_values.to('cuda'),
            y_values=batch.y_values.to('cuda'),
            target_x=batch.target_x.to('cuda'),
            target_y=batch.target_y.to('cuda'),
            context_x=batch.context_x.to('cuda'),
            context_y=batch.context_y.to('cuda'))


def load_FreyFace(img_height=28, img_width=20, img_channels=1,
                  shuffle=True, train_percent=0.75):
    ff = io.loadmat('./data/frey_rawface.mat')
    ff = ff["ff"].T.reshape((-1, img_channels, img_height, img_width))
    ff = ff.astype('float32')/255.
    ff_torch = torch.from_numpy(ff)
    if shuffle:
        shuffled_index = torch.randperm(len(ff_torch))
        ff_torch = ff_torch[shuffled_index]
    train_data = ff_torch[:int(len(ff_torch)*train_percent)]
    test_data = ff_torch[int(len(ff_torch)*train_percent):]
    return {'train': train_data, 'test': test_data}


def img_to_task(img,
                context_num=None,
                max_context_num=None,
                target_all=False,
                device=None):

    B, C, H, W = img.shape
    num_pixels = H*W
    img = img.view(B, C, -1)

    device = img.device if device is None else device

    max_context_num = max_context_num or num_pixels//2
    context_num = context_num or \
            torch.randint(low=3, high=max_context_num, size=[1]).item()
    target_num = num_pixels

    idxs = torch.rand(B, num_pixels).argsort(-1).to(img.device)
    x1, x2 = idxs//W, idxs%W
    x_values = torch.stack([
        2*x1.float()/(H-1) - 1,
        2*x2.float()/(W-1) - 1], -1).to(device)
    y_values = (torch.gather(img, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
            .transpose(-2, -1) - 0.5).to(device)

    context_x = x_values[:,:context_num]
    target_x = x_values[:,context_num:]
    context_y = y_values[:,:context_num]
    target_y = y_values[:,context_num:]

    return Batch_Description(
            x_values=x_values, y_values=y_values,
            target_x=target_x, target_y=target_y,
            context_x=context_x, context_y=context_y)


def coord_to_img(x, y, shape):
    x = x.cpu()
    y = y.cpu()
    B = x.shape[0]
    C, H, W = shape

    I = torch.zeros(B, 3, H, W)
    I[:,0,:,:] = 0.61
    I[:,1,:,:] = 0.55
    I[:,2,:,:] = 0.71

    x1, x2 = x[...,0], x[...,1]
    x1 = ((x1+1)*(H-1)/2).round().long()
    x2 = ((x2+1)*(W-1)/2).round().long()
    for b in range(B):
        for c in range(3):
            I[b,c,x1[b],x2[b]] = y[b,:,min(c,C-1)]

    return I


def task_to_img(xc, yc, xt, yt, shape):
    xc = xc.cpu()
    yc = yc.cpu()
    xt = xt.cpu()
    yt = yt.cpu()

    B = xc.shape[0]
    C, H, W = shape

    xc1, xc2 = xc[...,0], xc[...,1]
    xc1 = ((xc1+1)*(H-1)/2).round().long()
    xc2 = ((xc2+1)*(W-1)/2).round().long()

    xt1, xt2 = xt[...,0], xt[...,1]
    xt1 = ((xt1+1)*(H-1)/2).round().long()
    xt2 = ((xt2+1)*(W-1)/2).round().long()

    task_img = torch.zeros(B, 3, H, W).to(xc.device)
    task_img[:,2,:,:] = 1.0
    task_img[:,1,:,:] = 0.4
    for b in range(B):
        for c in range(3):
            task_img[b,c,xc1[b],xc2[b]] = yc[b,:,min(c,C-1)] + 0.5
    task_img = task_img.clamp(0, 1)

    completed_img = task_img.clone()
    for b in range(B):
        for c in range(3):
            completed_img[b,c,xt1[b],xt2[b]] = yt[b,:,min(c,C-1)] + 0.5
    completed_img = completed_img.clamp(0, 1)

    return task_img, completed_img


class FreyFace(Dataset):
    def __init__(self, data, transform=None):
        self._data = data
        self._transform = transform

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self._data[idx]
        if self._transform:
            image = self._transform(image)
        return (image, idx)


