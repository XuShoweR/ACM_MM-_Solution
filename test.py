import warnings
warnings.filterwarnings("ignore")

from include import *
from datasets import Dataset_
from models import *
from utils import *
import argparse
from tqdm import tqdm
import torchvision

def main(config):
    if config.model == 'res50':
        net = res50()
    elif config.model == 'se50':
        net = se50()
    elif config.model == 'res34':
        net = res34()
    elif config.model == 'se101':
        net = se101()
    elif config.model == 'se154':
        net = se154()
    elif config.model =='densenet201':
        net = torchvision.models.densenet201(pretrained=True)
        net.classifier = nn.Linear(net.classifier.in_features, 5)
    elif config.model =='resnet152':
        net = torchvision.models.resnet152(num_classes=1000, pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, 5)

    for fold in range(5):
        # if fold < 2:
        #   continue
        print('fold_{}'.format(fold))
        pretrained_dict = torch.load('../ckpt_extra/{}_4/{}_{}'.format(config.model, config.model_name, fold))
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.cuda()
        net.eval()

        test_dataset = Dataset_('test', image_size=(config.image_h, config.image_w), is_pseudo=False, tta=False, fold=fold)
        test_loader  = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size, num_workers=14, pin_memory=True)
        test_dataset_tta1 = Dataset_('test', image_size=(config.image_h, config.image_w), is_pseudo=False, tta=True, type=1, fold=fold)
        test_loader_tta1  = DataLoader(test_dataset_tta1, shuffle=False, batch_size=config.batch_size, num_workers=14, pin_memory=True)
        test_dataset_tta2 = Dataset_('test', image_size=(config.image_h, config.image_w), is_pseudo=False, tta=True, type=2, fold=fold)
        test_loader_tta2  = DataLoader(test_dataset_tta2, shuffle=False, batch_size=config.batch_size, num_workers=14, pin_memory=True)
        # test_dataset_tta3 = Dataset_('test', image_size=(config.image_h, config.image_w), is_pseudo=False, tta=True, type=3)
        # test_loader_tta3  = DataLoader(test_dataset_tta3, shuffle=False, batch_size=config.batch_size, num_workers=14, pin_memory=True)
        # test_dataset_tta4 = Dataset_('test', image_size=(config.image_h, config.image_w), is_pseudo=False, tta=True, type=4)
        # test_loader_tta4  = DataLoader(test_dataset_tta4, shuffle=False, batch_size=config.batch_size, num_workers=14, pin_memory=True)
        # test_dataset_tta5 = Dataset_('test', image_size=(config.image_h, config.image_w), is_pseudo=False, tta=True, type=5)
        # test_loader_tta5  = DataLoader(test_dataset_tta5, shuffle=False, batch_size=config.batch_size, num_workers=14, pin_memory=True)

        with torch.no_grad():

            probs1 = []
            probs2 = []
            probs3 = []
            probs4 = []
            probs5 = []
            probs6 = []
            for i,(id, input) in enumerate(tqdm(test_loader)):
                input = input.cuda()
                input = to_var(input)
                logit = net(input)
                prob    = F.softmax(logit)
                probs1 += prob.data.cpu().numpy().tolist()
            for i,(id, input) in enumerate(tqdm(test_loader_tta1)):
                input = input.cuda()
                input = to_var(input)
                logit = net(input)
                prob    = F.softmax(logit)
                probs2 += prob.data.cpu().numpy().tolist()
            for i,(id, input) in enumerate(tqdm(test_loader_tta2)):
                input = input.cuda()
                input = to_var(input)
                logit = net(input)
                prob    = F.softmax(logit)
                probs3 += prob.data.cpu().numpy().tolist()
            # for i,(id, input) in enumerate(tqdm(test_loader_tta3)):
            #     input = input.cuda()
            #     input = to_var(input)
            #     logit = net(input)
            #     prob    = F.softmax(logit)
            #     probs4 += prob.data.cpu().numpy().tolist()
            # for i,(id, input) in enumerate(tqdm(test_loader_tta4)):
            #     input = input.cuda()
            #     input = to_var(input)
            #     logit = net(input)
            #     prob    = F.softmax(logit)
            #     probs5 += prob.data.cpu().numpy().tolist()
            # for i,(id, input) in enumerate(tqdm(test_loader_tta5)):
            #     input = input.cuda()
            #     input = to_var(input)
            #     logit = net(input)
            #     prob    = F.softmax(logit)
            #     probs6 += prob.data.cpu().numpy().tolist()
            probs1 = np.array(probs1)
            probs2 = np.array(probs2)
            probs3 = np.array(probs3)
            # probs4 = np.array(probs4)
            # probs5 = np.array(probs5)
            # probs6 = np.array(probs6)
            probs = (probs1 + probs2 + probs3) / 3.0
            # probs = (probs1 + probs2 + probs3 + probs4 + probs5 + probs6) / 6.0
        np.save('../probs4_with_extra/{}_fold_{}.npy'.format(config.model, fold), probs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = 0)
    parser.add_argument('--model', type=str, default='res34')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)

    parser.add_argument('--image_h', type=int, default=512)
    parser.add_argument('--image_w', type=int, default=512)

    parser.add_argument('--s1', type=float, default=64.0)
    parser.add_argument('--m1', type=float, default=0.0)
    parser.add_argument('--s2', type=float, default=16.0)

    parser.add_argument('--focal_w', type=float, default=1.0)
    parser.add_argument('--softmax_w', type=float, default=1.0)
    parser.add_argument('--triplet_w', type=float, default=1.0)

    # parser.add_argument('--mode', type=str, default='train', choices=['train', 'val','val_fold','test_classifier','test','test_fold'])
    # parser.add_argument('--pretrained_model', type=str, default=None)
    #
    parser.add_argument('--mode', type=str, default='test_classifier', choices=['train', 'val','val_fold','test_classifier','test','test_fold'])
    parser.add_argument('--pretrained_model', type=str, default='max_valid_model.pth')

    parser.add_argument('--iter_save_interval', type=int, default=5)
    parser.add_argument('--train_epoch', type=int, default=50)

    config = parser.parse_args()
    main(config)
