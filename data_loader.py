import torch
import torch.utils.data
import pandas as pd
import numpy as np

window_size = 40

num_train = 500
num_val = 250
num_test = 250
indices = [0, 250, 500, 750, 1000]


def df_to_list(df):
    df = df.reset_index(drop=True)

    data = []

    HM = torch.zeros(1, 7, window_size)
    for i in range(len(df)):
        row = df.loc[i].values
        gene_id = row[0]

        for j in range(7):
            HM[0][j] = torch.Tensor(list(map(float, row[window_size * j + 1: window_size * (j + 1) + 1])))

        ME = torch.zeros(2, 436)
        ME[0] = torch.Tensor(list(map(float, row[280+1: 280+436+1])))
        ME[1] = torch.Tensor(list(map(float, row[280+436+1: 280+436+436+1])))
        ME.transpose_(0, 1)

        ME_length = np.int(row[280+436+436+1])

        TF = torch.zeros(3, 1016)
        TF[0] = torch.Tensor(list(map(float, row[280+436+436+1+1:280+436+436+1+1+1016])))
        TF[1] = torch.Tensor(list(map(float, row[280+436+436+1+1+1016:280+436+436+1+1+1016+1016])))
        TF[2] = torch.Tensor(list(map(float, row[280+436+436+1+1+1016+1016:280+436+436+1+1+1016+1016+1016])))

        GE = np.long(row[-3])
        GE_value = np.float(row[-1])
        data.append((gene_id, HM.clone(), ME.clone(), ME_length, TF.clone(), GE, GE_value))
    return data


class EpigenomeDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = df_to_list(df)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def make_data_loader(cell_line, cross_validation_seed=2):
    print("reading and shuffling the file")
    df = pd.read_csv('./toy_dataset/' + cell_line + '_toy.csv', header=None, index_col=False).sample(frac=1, random_state=123).reset_index(drop=True)
    quarter = []

    for i in range(4):
        quarter.append(df.loc[indices[i]: indices[i+1] - 1])

    val_df = quarter[cross_validation_seed]
    test_df = quarter[(cross_validation_seed + 1) % 4]
    train_df = pd.concat((quarter[(cross_validation_seed + 2) % 4], quarter[(cross_validation_seed + 3) % 4]))

    print("making data loaders")
    train_loader = torch.utils.data.DataLoader(EpigenomeDataset(train_df), batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(EpigenomeDataset(val_df), batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(EpigenomeDataset(test_df), batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader
