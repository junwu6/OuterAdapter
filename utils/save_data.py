import numpy as np
import pickle


def save_data(dataloader, name):
    X, Y = [], []
    for data, target in dataloader:
        X.append(data.detach().cpu().numpy())
        Y.append(target.detach().cpu().numpy())
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    print(X.shape, Y.shape)

    data = {}
    data['X'] = X
    data['Y'] = Y

    with open("data_preprocessed/{}.pkl".format(name), "wb") as pkl_file:
        pickle.dump(data, pkl_file)
