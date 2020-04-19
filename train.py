import warnings
from MobileNet import MobileNetV2
import pretrainedmodels
import torchvision

warnings.filterwarnings("ignore")

from include import *
from datasets import Dataset_
from models import *
from utils import *
import argparse
from Nadam import Nadam
from losses import CrossEntropyLabelSmooth
import model

def softmax_loss(results, labels):
    # print(results.shape)
    # labels = labels.view(-1)
    # print(labels.shape)
    loss = F.cross_entropy(results, labels, reduce=True)
    return loss

CE_smooth = CrossEntropyLabelSmooth(num_classes=50)

def do_valid(net, valid_loader):
    valid_num  = 0
    truths   = []
    losses   = []
    corrects = []
    probs = []
    labels = []

    with torch.no_grad():
        for input, truth_ in valid_loader:
            input = input.cuda()
            truth_ = truth_.cuda().long()

            # input = to_var(input)
            # truth_ = to_var(truth_)

            logit = net(input)
            loss = softmax_loss(logit, truth_)

            probs.append(logit)
            labels.append(truth_)
            valid_num += len(input)

            loss_tmp = loss.data.cpu().numpy().reshape([1])
            losses.append(loss_tmp)
            truths.append(truth_.data.cpu().numpy())


    assert (valid_num == len(valid_loader.sampler))
    # ------------------------------------------------------
    loss = np.concatenate(losses,axis=0)
    loss = loss.mean()
    prob = torch.cat(probs)
    label = torch.cat(labels)

    _, precision = metric(prob, label)

    return loss, precision


def run_train(config, fold=0):
    if config.model == 'res50':
        net = res50()
    elif config.model == 'se50':
        net = se50()
    elif config.model == 'res34':
        net = res34()
    elif config.model == 'se101':
        net = se101()
    elif config.model == 'res34_atten':
        net = res34_attention_pool()
    elif config.model == 'se154':
        net = se154()
    elif config.model =='densenet201':
        net = torchvision.models.densenet201(pretrained=True)
        net.classifier = nn.Linear(net.classifier.in_features, 5)
    elif config.model =='resnet152':
        net = torchvision.models.resnet152(num_classes=1000, pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, 5)
    elif config.model =='densenet161':
        net = torchvision.models.densenet161(pretrained=True)
        net.classifier = nn.Linear(net.classifier.in_features, 5)
    elif config.model == 'inceptionv4':
        net = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        net.last_linear = nn.Linear(net.last_linear.in_features, 5)
        # IMAGE_SIZE = 299
    elif config.model == 'inceptionresnetv2':
        net = model.inceptionresnetv2_finetune(5)
    elif config.model == 'mobilenet':
        net = MobileNetV2(n_class=1000)
        net.load_state_dict(torch.load("../mobilenet_v2.pth.tar"))
        net.classifier = nn.Linear(net.last_channel, 5)
    #net.load_state_dict(torch.load('../ckpt/res34/model_6'))
    net = nn.DataParallel(net)
    net.cuda()

    train_dataset = Dataset_('train', image_size=(config.image_h, config.image_w), is_pseudo=True, fold=fold)
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=14, pin_memory=True)
    valid_dataset = Dataset_('val', image_size=(config.image_h, config.image_w), is_pseudo=False, fold=fold)
    valid_loader  = DataLoader(valid_dataset, shuffle=False, batch_size=config.batch_size, num_workers=14, pin_memory=True)



    if not os.path.isdir('../logs_extra/{}'.format(config.model)):
        os.mkdir('../logs_extra/{}'.format(config.model))
    log = open('../logs_extra/{}'.format(config.model)+'/log.train.txt', mode='a')
    log.write('\t__file__     = %s\n')
    log.write('\tout_dir      = %s\n')
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    assert(len(train_dataset)>=config.batch_size)
    log.write('batch_size = %d\n'%(config.batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    # if initial_checkpoint is not None:
    #     log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    #     net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    #     print('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    log.write('%s\n'%(type(net)))
    log.write('\n')

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr)
    #optimizer = Nadam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr)
    iter_smooth = 20
    start_iter = 0

    log.write('\n')
    ## start training here! ##############################################
    log.write('** top_n step 100,60,60,60 **\n')
    log.write('** start training here! **\n')
    log.write('                      |---- VALID ----|- TRAIN/BATCH -|         \n')
    log.write('rate    iter   epoch  | loss   acc-1  | loss   acc-1  |  time   \n')
    log.write('--------------------------------------------------------------\n')

    print('** start training here! **\n')
    print('                      |----- VALID ----|--TRAIN/BATCH -|         \n')
    print('rate    iter   epoch  | loss   acc-1   | loss   acc-1  |  time   \n')
    print('---------------------------------------------------------------\n')

    def adjust_lr(optimizer, ep):
        if ep < 12:
            lr = 3e-4
        elif ep < 16:
            lr = 1e-4
        elif ep < 19:
            lr = 1e-5
        else:
            lr = 1e-6
        for p in optimizer.param_groups:
            p['lr'] = lr
        return lr
    # def get_lr(ep):
    #     if ep < 12:
    #         lr = 3e-4
    #     elif ep < 16:
    #         lr = 1e-4
    #     elif ep < 19:
    #         lr = 1e-5
    #     else:
    #         lr = 1e-6
    #     return lr
    # def adjust_lr_ep(optimizer, ep):
    #     lr = config.lr * ep / 10

    # def adjust_lr(optimizer, lr):
    #     for p in optimizer.param_groups:
    #         p['lr'] = lr

    i    = 0
    start = timer()
    max_valid = 0.
    patience = 0
    max_lr_change = 3
    lrs = [3e-4, 1e-4, 1e-5, 1e-6]
    k = 0
    for epoch in range(config.train_epoch):
        train_loss = []
        train_acc  = []
        valid_loss = []
        valid_acc  = []
        rate = adjust_lr(optimizer, epoch)
        # rate = get_lr(epoch)
        optimizer.zero_grad()

        # rate, hard_ratio = adjust_lr_and_hard_ratio(optimizer, epoch + 1)
        # rate = lrs[k]

        for input, truth_ in train_loader:
            iter = i + start_iter
            # one iteration update  -------------
            net.train()
            input = input.cuda()
            truth_ = truth_.cuda().long()
            # print(truth_)

            # input = to_var(input)
            # truth_ = to_var(truth_)

            logit = net(input)

            # loss_focal = focal_OHEM(logit, truth_,truth, hard_ratio)* config.focal_w
            loss_softmax = softmax_loss(logit, truth_) * config.softmax_w
            # loss_triplet = TripletLoss(margin=0.3)(feas, truth_, normalize_feature=True) * config.triplet_w

            loss = loss_softmax
            _, precision = metric(logit, truth_)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())
            train_acc.append(precision.item())


        train_loss = np.mean(train_loss)
        train_acc  = np.mean(train_acc)

        net.eval()
        valid_loss, val_acc = do_valid(net, valid_loader)
        net.train()

        print('%0.6f %5.1f %6.1f | %0.3f  %0.3f%s  | %0.3f  %0.3f  | %s' % (\
                    rate, iter, epoch,
                    valid_loss, val_acc,' ',
                    train_loss, train_acc,
                    time_to_str((timer() - start),'min')))
        log.write('%0.6f %5.1f %6.1f | %0.3f  %0.3f%s  | %0.3f  %0.3f  | %s' % (\
                    rate, iter, epoch,
                    valid_loss, val_acc,' ',
                    train_loss, train_acc,
                    time_to_str((timer() - start),'min')))

        log.write('\n')

        if max_valid < val_acc:
            patience = 0
            max_valid = val_acc
            # print('save max valid!!!!!! : ' + str(max_valid))
            # log.write('save max valid!!!!!! : ' + str(max_valid))
            # log.write('\n')
            if not os.path.isdir('../ckpt_extra/{}_4/'.format(config.model)):
                os.mkdir('../ckpt_extra/{}_4/'.format(config.model))
            torch.save(net.state_dict(), '../ckpt_extra/{}_4/{}_{}'.format(config.model, config.model_name, fold))
        # else:
        #     patience += 1
        #     if patience == 4:
        #         k += 1
        #         if k == 4:
        #             break
        #         # adjust_lr(optimizer, ep)
        #         adjust_lr(optimizer, lrs[k])
        #         net.load_state_dict(torch.load('../ckpt/{}_4/{}_{}'.format(config.model, config.model_name, fold)))
        #         patience = 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = 0)
    parser.add_argument('--model', type=str, default='res34')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--log_name', type=str, default='log1')
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
    # parser.add_argument('--fold', type=int, default=4, required=True)

    parser.add_argument('--iter_save_interval', type=int, default=5)
    parser.add_argument('--train_epoch', type=int, default=22)

    config = parser.parse_args()

    for fold in range(5):
    # fold = config.fold
        print('fold_{}'.format(fold))
    # if fold > 0:
    #    break
        run_train(config, fold)
