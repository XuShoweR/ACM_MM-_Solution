import pickle




for fold in range(5):
    with open('../input/folds/train_ids_{}.pkl'.format(fold), 'rb') as f:
        train_ids = pickle.load(f)
    with open('../input/folds/val_ids_{}.pkl'.format(fold), 'rb') as f:
        val_ids = pickle.load(f)
    with open('../input/folds/train_targets_{}.pkl'.format(fold), 'rb') as f:
        train_targets = pickle.load(f)
    with open('../input/folds/val_targets_{}.pkl'.format(fold), 'rb') as f:
        val_targets = pickle.load(f)