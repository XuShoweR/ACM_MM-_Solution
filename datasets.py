from include import *
from imgaug import augmenters as iaa
import pickle
import random

## MY
# TRN_IMGS_DIR  = '/home/joya2/gary/cassava/input/'
# TST_IMGS_DIR  = '/home/joya2/gary/idesign/input/test/test/'
# INPUT_DIR = '/home/joya2/gary/idesign/input/'

# with open('/home/joya2/gary/idesign/input/train_ids.pkl', 'rb') as f:
#     train_ids = pickle.load(f)
# with open('/home/joya2/gary/idesign/input/val_ids.pkl', 'rb') as f:
#     val_ids = pickle.load(f)
# with open('/home/joya2/gary/idesign/input/train_targets.pkl', 'rb') as f:
#     train_targets = pickle.load(f)
# with open('/home/joya2/gary/idesign/input/val_targets.pkl', 'rb') as f:
#     val_targets = pickle.load(f)

# TAO
# TRN_IMGS_DIR = '/media/st/SSD/gary/idesign/input/train/designer_image_train_v2_cropped/'
# TST_IMGS_DIR  = '/media/st/SSD/gary/idesign/input/test/test/'
# INPUT_DIR = '/media/st/SSD/gary/idesign/input/'
#
# with open('{}train_ids.pkl'.format(INPUT_DIR), 'rb') as f:
#     train_ids = pickle.load(f)
# with open('{}val_ids.pkl'.format(INPUT_DIR), 'rb') as f:
#     val_ids = pickle.load(f)
# with open('{}train_targets.pkl'.format(INPUT_DIR), 'rb') as f:
#     train_targets = pickle.load(f)
# with open('{}val_targets.pkl'.format(INPUT_DIR), 'rb') as f:
#     val_targets = pickle.load(f)

## test_ids
with open('../input/test_ids.pkl', 'rb') as f:
    test_ids = pickle.load(f)


## extra_ids
# with open('../input/extra_iiid.pkl', 'rb') as f:
#     extra_iiid = pickle.load(f)
# with open('../input/ex_targets.pkl', 'rb') as f:
#     extra_targets = pickle.load(f)

# test_ids = np.load('../input/test_ids.npy'.format(INPUT_DIR))

def random_cropping(image, ratio=0.8, is_random=True):
    height, width, _ = image.shape
    target_h = int(height * ratio)
    target_w = int(width * ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = (width - target_w) // 2
        start_y = (height - target_h) // 2

    zeros = image[start_y:start_y + target_h, start_x:start_x + target_w, :]
    zeros = cv2.resize(zeros, (width, height))
    return zeros


def random_erase(image, p=0.5):
    if random.random() < p:
        width, height, d = image.shape
        x = random.randint(0, width)
        y = random.randint(0, height)
        b_w = random.randint(10, 20)
        b_h = random.randint(5, 10)
        image[x:x + b_w, y:y + b_h] = 0
    return image


def hideseek(img):
    # get width and height of the image\
    img = img.copy()
    s = img.shape
    wd = s[0]
    ht = s[1]

    # possible grid size, 0 means no hiding
    #     grid_sizes=[0,16,32,44,56]

    # hiding probability
    hide_prob = 0.33

    # randomly choose one grid size
    #     grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]
    grid_size = 44
    # hide the patches
    if (grid_size > 0):
        for x in range(0, wd, grid_size):
            for y in range(0, ht, grid_size):
                x_end = min(wd, x + grid_size)
                y_end = min(ht, y + grid_size)
                if (random.random() <= hide_prob):
                    img[x:x_end, y:y_end, :] = 0
    return img


def do_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)  # apply gamma correction using the lookup table


def do_contrast(image, alpha=1.0):
    image = image.astype(np.float32)
    gray = image * np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    image = alpha * image + gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_brightness_multiply(image, alpha=1):
    image = image.astype(np.float32)
    image = alpha * image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_brightness_shift(image, alpha=0.125):
    image = image.astype(np.float32)
    image = image + alpha * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


class Dataset_(Dataset):
    def __init__(self, mode, image_size=(128, 128),
                 augment=None,
                 is_pseudo=False, tta=False, type=1, fold=0):

        super(Dataset_, self).__init__()
        self.mode = mode
        self.augment = augment
        self.is_pseudo = is_pseudo
        self.fold = fold
        # self.train_image_path = TRN_IMGS_DIR
        # self.val_image_path = TRN_IMGS_DIR
        # self.test_image_path = TST_IMGS_DIR
        with open('../input/folds/train_ids_{}.pkl'.format(self.fold), 'rb') as f:
            self.train_ids = pickle.load(f)
        with open('../input/folds/val_ids_{}.pkl'.format(self.fold), 'rb') as f:
            self.val_ids = pickle.load(f)
        with open('../input/folds/train_targets_{}.pkl'.format(self.fold), 'rb') as f:
            self.train_targets = pickle.load(f)
        with open('../input/folds/val_targets_{}.pkl'.format(self.fold), 'rb') as f:
            self.val_targets = pickle.load(f)

        # if self.is_pseudo:
        #     ## extra_ids
        #     # with open('../input/extra_iiid.pkl', 'rb') as f:
        #     #     extra_iiid = pickle.load(f)
        #     # with open('../input/ex_targets.pkl', 'rb') as f:
        #     #     extra_targets = pickle.load(f)
        #     with open('../input/extra_iiid_{}.pkl'.format(self.fold), 'rb') as f:
        #         extra_iiid = pickle.load(f)
        #     with open('../input/ex_targets{}.pkl'.format(self.fold), 'rb') as f:
        #         extra_targets = pickle.load(f)
        #     with open('../input/t_train_ids_{}.pkl'.format(self.fold), 'rb') as f:
        #         extra_iiid2 = pickle.load(f)
        #     with open('../input/t_train_targets_{}.pkl'.format(self.fold), 'rb') as f:
        #         extra_targets2 = pickle.load(f)
        #     # with open('../input/test_iid.pkl', 'rb') as f:
        #     #     extra_iiid2 = pickle.load(f)
        #     # with open('../input/test_targets.pkl', 'rb') as f:
        #     #     extra_targets2 = pickle.load(f)
        #     self.train_ids = np.concatenate((self.train_ids, extra_iiid, extra_iiid2))
        #     self.train_targets = np.concatenate((self.train_targets, extra_targets, extra_targets2))

        # if self.is_pseudo:
        #     self.train_ids = np.concatenate((self.train_ids, extra_iiid))
        #     self.train_targets = np.concatenate((self.train_targets, extra_targets))

        self.image_size = image_size
        self.fold_index = None
        if mode == 'train':
            self.seq = iaa.Sequential([iaa.Fliplr(0.5)])
            self.seq_ud = iaa.Sequential([iaa.Flipud(0.5)])
            self.seq_shear = iaa.Sequential([iaa.Affine(shear=(-30, 30), mode='wrap')])
            self.seq_trans = iaa.Sequential([iaa.Affine(translate_percent=(-0.15, 0.15), mode='wrap')])
            self.seq_rotate = iaa.Sequential([iaa.Affine(rotate=(-180, 180), mode='wrap')])

        #     self.seq_affine = iaa.Sequential([
        #     iaa.Affine(rotate= (-15, 15),
        #                shear = (-15, 15),
        #                mode='edge'),

        #     iaa.SomeOf((0, 2),
        #                [
        #                    iaa.GaussianBlur((0, 1.5)),
        #                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
        #                    iaa.AddToHueAndSaturation((-5, 5)),  # change hue and saturation
        #                ],
        #                random_order=True
        #                )
        # ])
        # self.seq_affine = iaa.Sequential([iaa.Affine(rotate=(-60, 60))])
        if mode == 'test':
            # with open('../input/t_val_ids_{}.pkl'.format(self.fold), 'rb') as f:
            #     self.test_ids = pickle.load(f)
            self.test_dir = '../input/test/0/'
            # self.test_dir = '../input/extraimages/'
        #     self.test_list = os.listdir(self.test_dir)
        if mode == 'test' and tta:
            self.seq = iaa.Sequential([iaa.Fliplr(1.0)])
            self.seq_ud = iaa.Sequential([iaa.Flipud(1.0)])
            self.seq_rotate_1 = iaa.Sequential([iaa.Affine(rotate=180, mode='wrap')])
            self.seq_rotate_2 = iaa.Sequential([iaa.Affine(rotate=90, mode='wrap')])
            self.seq_rotate_3 = iaa.Sequential([iaa.Affine(rotate=-90, mode='wrap')])
        self.tta = tta
        self.type = type
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode

        if self.mode == 'train':
            print(len(self.train_ids))
            self.num_data = len(self.train_ids)
            print('set dataset mode: train')

        elif self.mode == 'val':
            print(len(self.val_ids))
            self.num_data = len(self.val_ids)
            print('set dataset mode: val')

        elif self.mode == 'test':
            self.test_list = test_ids
            # self.test_list = self.test_ids
            self.num_data = len(self.test_list)
            print('set dataset mode: test')

        print('data num: ' + str(self.num_data))

    def __getitem__(self, index):

        if self.mode == 'train':
            image_tmp = self.train_ids[index]
            label = self.train_targets[index]
            # image_tmp, label = self.train_list[index]
            image = cv2.imread('{}'.format(image_tmp), 1)

        if self.mode == 'val':
            image_tmp = self.val_ids[index]
            label = self.val_targets[index]
            image = cv2.imread('{}'.format(image_tmp), 1)

        if self.mode == 'test':
            # image_path = os.path.join(self.test_dir, self.test_list[index])
            image_path = self.test_list[index]
            image = cv2.imread(self.test_dir + image_path, 1)

            image_id = self.test_list[index]

            image = random_cropping(image, is_random=False)
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

            if self.tta and self.type == 1:
                image = self.seq.augment_image(image)
            elif self.tta and self.type == 2:
                image = self.seq_ud.augment_image(image)
            elif self.tta and self.type == 3:
                image = self.seq_rotate_1.augment_image(image)
            elif self.tta and self.type == 4:
                image = self.seq_rotate_2.augment_image(image)
            elif self.tta and self.type == 5:
                image = self.seq_rotate_3.augment_image(image)

            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([3, self.image_size[0], self.image_size[1]])
            image = image / 255.0

            return image_id, torch.FloatTensor(image)

        if self.mode == 'train':
            image = random_cropping(image, is_random=True)
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image = self.seq.augment_image(image)
            image = self.seq_ud.augment_image(image)
            if random.randint(0, 1) == 0:
                image = self.seq_rotate.augment_image(image)
            # if random.randint(0, 1) == 0:
            #     image = self.seq_shear.augment_image(image)
            # if random.randint(0, 1) == 0:
            #     choice = random.randint(0, 1)
            #     if choice == 0:
            #         image = self.seq_rotate.augment_image(image)
            #     else:
            #         image = self.seq_shear.augment_image(image)   
            # if random.randint(0, 1) == 0:
            #     image = self.seq_trans.augment_image(image)

        if self.mode == 'val':
            image = random_cropping(image, is_random=False)
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        # elif self.mode == 'val':
        #     image = aug_image(image, is_infer=True, augment=self.augment)

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image.reshape([-1, self.image_size[0], self.image_size[1]])
        image = image / 255.0

        return torch.FloatTensor(image), label

    def __len__(self):
        return self.num_data
