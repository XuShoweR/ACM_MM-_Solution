from include import *
from torch.autograd import Variable

def save(list_or_dict,name):
    f = open(name, 'w')
    f.write(str(list_or_dict))
    f.close()

def load(name):
    f = open(name, 'r')
    a = f.read()
    tmp = eval(a)
    f.close()
    return tmp

def dot_numpy(vector1 , vector2,emb_size = 512):
    vector1 = vector1.reshape([-1, emb_size])
    vector2 = vector2.reshape([-1, emb_size])
    vector2 = vector2.transpose(1,0)

    cosV12 = np.dot(vector1, vector2)
    return cosV12

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    loss = F.cross_entropy(logit, truth, reduce=is_average)
    return loss

def softmax_add_newwhale(logit, truth):
    indexs_NoNew = (truth != 5004).nonzero().view(-1)
    indexs_New = (truth == 5004).nonzero().view(-1)

    logits_NoNew = logit[indexs_NoNew]
    truth_NoNew = truth[indexs_NoNew]

    logits_New = logit[indexs_New]
    print(logits_New.size())

    if logits_NoNew.size()[0]>0:
        loss = nn.CrossEntropyLoss(reduce=True)(logits_NoNew, truth_NoNew)
    else:
        loss = 0

    if logits_New.size()[0]>0:
        logits_New = torch.softmax(logits_New,1)
        logits_New = logits_New.topk(1,1,True,True)[0]
        target_New = torch.zeros_like(logits_New).float().cuda()
        loss += nn.L1Loss()(logits_New, target_New)

    return loss

# def bce_criterion(logit, truth, top_n = None):
#
#     if top_n is None:
#         loss = F.binary_cross_entropy_with_logits(logit, truth, reduce=True)
#         return loss
#     else:
#         loss = F.binary_cross_entropy_with_logits(logit, truth, reduce=False)
#         value,  index= loss.topk(top_n, dim=1, largest=True, sorted=True)
#         return value.mean()


def metric(logit, truth, is_average=True, is_prob = False):
    if is_prob:
        prob = logit
    else:
        prob = F.softmax(logit, 1)

    value, top = prob.topk(5, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))

    if is_average==True:
        # top-3 accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct/len(truth)

        top = [correct[0],
               correct[0] + correct[1],
               correct[0] + correct[1] + correct[2],
               correct[0] + correct[1] + correct[2] + correct[3],
               correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]

        precision = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5

        return precision, top[0]
    else:
        return correct

def metric_for_5005(prob, label, thres = 0.5):

    shape = prob.shape
    prob_5005 = np.ones([shape[0], shape[1] + 1]) * thres
    prob_5005[:, :5004] = prob

    precision , top5 = top_n_np(prob_5005, label)
    return  precision, top5

# def metric(prob, label):
#     precision , top5 = top_n_np(prob, label)
#     return  precision, top5

def metric_for_4flip(prob, label, thres = 0.5):

    shape = prob.shape
    prob_5005 = np.ones([shape[0], shape[1] + 1]) * thres
    prob_5005[:, :5004*4] = prob

    precision , top5 = top_n_np(prob_5005, label)
    return  precision, top5


def top_n_np(preds, labels):
    n = 5
    predicted = np.fliplr(preds.argsort(axis=1)[:, -n:])
    top5 = []

    re = 0
    for i in range(len(preds)):
        predicted_tmp = predicted[i]
        labels_tmp = labels[i]
        for n_ in range(5):
            re += np.sum(labels_tmp == predicted_tmp[n_]) / (n_ + 1.0)

    re = re / len(preds)

    for i in range(n):
        top5.append(np.sum(labels == predicted[:, i])/ (1.0*len(labels)))

    return re, top5

def metric_binary(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(2, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))
    correct = correct.float().sum(0, keepdim=False)
    correct = correct / len(truth)
    return correct[0]

def metric_bce(logit, truth):
    prob = F.sigmoid(logit)
    prob[prob > 0.5] = 1
    prob[prob < 0.5] = 0
    correct =   prob.eq(truth.view(-1, 1).expand_as(prob))
    correct = correct.float().sum(0, keepdim=False)
    correct = correct/len(truth)
    return correct

def do_valid_siamese(model, net, valid_loader, criterion ):
    valid_num  = 0
    probs    = []
    truths   = []
    losses   = []
    corrects = []

    for input_A,input_B, truth in valid_loader:
        inputA = to_var(input_A)
        inputB = to_var(input_B)
        truth = to_var(truth)

        fea_A = model.forward(inputA)
        fea_B = model.forward(inputB)

        fea = torch.cat([fea_A, fea_B], dim=1)
        logit = net.forward(fea)
        loss = criterion(logit, truth)
        correct = metric_bce(logit, truth)

        valid_num += len(input_A)
        # probs.append(prob.data.cpu().numpy())
        losses.append(loss.data.cpu().numpy().reshape([-1]))
        corrects.append(correct.data.cpu().numpy().reshape([-1]))
        truths.append(truth.data.cpu().numpy())


    assert(valid_num == len(valid_loader.sampler))
    #------------------------------------------------------
    correct = np.concatenate(corrects)
    loss    = np.concatenate(losses)
    loss    = loss.mean()
    precision = correct.mean()

    valid_loss = np.array([
        loss, 0.0, 0.0, precision
    ])
    return valid_loss


def do_valid( net, valid_loader, criterion ):
    valid_num  = 0
    probs    = []
    truths   = []
    losses   = []
    corrects = []

    for input, truth in valid_loader:
        input = input.cuda()
        truth = truth.cuda()

        input = to_var(input)
        truth = to_var(truth)

        logit, _   = net(input, truth, is_infer = True)
        prob    = F.softmax(logit,1)

        loss    = criterion(logit, truth, False)
        correct = metric(logit, truth, False)

        valid_num += len(input)
        probs.append(prob.data.cpu().numpy())
        losses.append(loss.data.cpu().numpy())
        corrects.append(correct.data.cpu().numpy())
        truths.append(truth.data.cpu().numpy())


    assert(valid_num == len(valid_loader.sampler))
    #------------------------------------------------------
    prob    = np.concatenate(probs)
    correct = np.concatenate(corrects)
    truth   = np.concatenate(truths).astype(np.int32).reshape(-1,1)
    loss    = np.concatenate(losses)
    #---
    #top = np.argsort(-predict,1)[:,:3]

    loss    = loss.mean()
    correct = correct.mean(0)
    top = [correct[0],
           correct[0]+correct[1] ,
           correct[0]+correct[1]+correct[2],
           correct[0]+correct[1]+correct[2]+correct[3],
           correct[0] + correct[1] + correct[2] + correct[3]+ correct[4]]


    precision = correct[0]/1 + correct[1]/2 + correct[2]/3 + correct[3]/4 + correct[4]/5

    #----
    valid_loss = np.array([
        loss, top[0], top[4], precision
    ])
    return valid_loss




def load_train_map(train_image_list_path = r'/data2/shentao/Projects/Kaggle_Whale/image_list/train_image_list.txt'):
    f = open(train_image_list_path, 'r')
    lines = f.readlines()
    f.close()

    label_dict = {}
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        img_name = line[0]
        index = int(line[1])
        id = line[2]
        label_dict[img_name] = [index,id]

    return label_dict

#

def prob_to_csv_top5(prob, key_id, name):
    CLASS_NAME,_ = load_CLASS_NAME()

    prob = np.asarray(prob)
    print(prob.shape)

    top = np.argsort(-prob,1)[:,:5]
    word = []
    index = 0

    rs = []

    for (t0,t1,t2,t3,t4) in top:
        word.append(
            CLASS_NAME[t0] + ' ' + \
            CLASS_NAME[t1] + ' ' + \
            CLASS_NAME[t2])

        top_k_label_name = r''

        label = CLASS_NAME[t0]
        score = prob[index][t0]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t1]
        score = prob[index][t1]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t2]
        score = prob[index][t2]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t3]
        score = prob[index][t3]
        top_k_label_name += label + ' ' + str(score) + ' '

        label = CLASS_NAME[t4]
        score = prob[index][t4]
        top_k_label_name += label + ' ' + str(score) + ' '

        # print(top_k_label_name)
        rs.append(top_k_label_name)
        index += 1
        # break

    pd.DataFrame({'key_id':key_id, 'word':rs}).to_csv( '{}.csv'.format(name), index=None)


if __name__ == '__main__':
    dict = load_train_map()
    print(dict)