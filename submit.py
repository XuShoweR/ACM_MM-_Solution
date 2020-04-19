import argparse

import numpy as np
import pandas as pd
import pickle
import os

# parser = argparse.ArgumentParser()
# parser.add_argument('--mode', type=str, )
# args = parser.parse_args()


# model = args.mode
id_dict = {0:'cbb', 1:'cbsd', 2:'cgm', 3:'cmd', 4:'healthy'}

with open('../input/test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)

# for i in range(5):
#     prob = np.load('../probs4/{}_fold_{}.npy'.format(model, i))
#     if i == 0:
#         se50 = prob
#     else:
#         se50 += prob
#
# se50 = se50 / 5.0

# for i, file in enumerate(os.listdir('../probs4_with_extra')):
#     if file.startswith('resnet'):
#         continue
#     print(file)
#     prob = np.load(os.path.join('../probs4_with_extra', file))
#     if i == 0:
#         se50 = prob
#     else:
#         se50 += prob

i = 0

for j, file in enumerate(os.listdir('../probs4')):
    if file.startswith('se101'):
        print(file)
        prob = np.load(os.path.join('../probs4', file))
        if i == 0:
            se50 = prob
            i += 1
        else:
            se50 += prob

se50 = se50 / 5.0


top = np.argsort(-se50,1)[:,0]

pred = []
for t in top:
    pred.append(id_dict[t])



pd.DataFrame({'Category':pred, 'Id':test_ids}).to_csv( './sub/{}.csv'.format('ensemble_with_extra_se50se101densenet'), index=None)