import os
import torch
import collections
import numpy as np
import pandas as pd


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


def load_data(train_percent=0.75):
    lane2 = pd.read_csv('./data/lane2.csv', index_col=0).to_numpy()
    lane3 = pd.read_csv('./data/lane3.csv', index_col=0).to_numpy()

    np.random.shuffle(lane2)
    np.random.shuffle(lane3)

    flow = np.expand_dims(np.stack([lane2[:, 0], lane3[:, 0]]),
                      axis=-1)
    speed = np.expand_dims(np.stack([lane2[:, 1], lane3[:, 1]]),
                       axis=-1)
    
    flow, speed = torch.FloatTensor(flow), torch.FloatTensor(speed)
    flow, speed = 2 * flow / torch.max(flow) - 1, speed / torch.max(speed)
    
    _, num_samples, _ = flow.shape
    num_train_samples = int(num_samples * train_percent)
    
    train_data = {'flow': flow[:, :num_train_samples, :],
                  'speed': speed[:, :num_train_samples, :]
                  }
    test_data = {'flow': flow[:, num_train_samples:, :],
                 'speed': speed[:, num_train_samples:, :]
                 }

    return train_data, test_data


class SpeedFlow(object):
    def __init__(self, train_data, test_data):
        self._train_data = train_data
        self._test_data = test_data

    def sample(self, min_num_context=None, testing=False, device='cpu'):
        _, num_train_samples, _ = self._train_data['flow'].shape
        _, num_test_samples, _ = self._test_data['flow'].shape

        if testing is True:
            context_x = self._train_data['flow']
            context_y = self._train_data['speed']
            target_x = self._test_data['flow']
            target_y = self._test_data['speed']
            x_values = torch.cat((context_x, target_x), dim=1)
            y_values = torch.cat((context_y, target_y), dim=1)

        else:
            min_num_context = min_num_context or 500
            num_context = torch.randint(low=min_num_context, high=num_train_samples-2, size=(1,))
            shuffled_index = torch.randperm(num_train_samples)

            x_values = self._train_data['flow'][:, shuffled_index, :]
            y_values = self._train_data['speed'][:, shuffled_index, :]

            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]
            target_x = x_values[:, num_context:, :]
            target_y = y_values[:, num_context:, :]

        x_values, y_values = x_values.to(device), y_values.to(device)
        context_x, context_y = context_x.to(device), context_y.to(device)
        target_x, target_y = target_x.to(device), target_y.to(device)

        return Batch_Description(
            x_values=x_values, y_values=y_values,
            target_x=target_x, target_y=target_y,
            context_x=context_x, context_y=context_y)


