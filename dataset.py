from torch.utils.data import Dataset
import random
import cv2, torch
import os
from pathlib import Path
import torchvision.transforms.v2 as transforms


class ISIC_aug_dataset(Dataset):

    def __init__(self, path:str, type:str, image_size = 256):
        '''
        path: augmented dataset root directory
        type: "train" or "test"
        image_size: 如果需要改变图像SIZE可以临时修改
        可以修改参数，将image_size改为transforms，并修改代码，导入指定的transform，不过需要注意，
        需要对图像和掩码做一致的transform，如带有随机变换务必先将图像和掩码拼接
        '''
        self.type = type
        data_path, gt_path = os.path.join(path, "ISIC_"+self.type+"_image"), os.path.join(path, "ISIC_"+self.type+"_groundtruth")
        images_list = list(Path(data_path).glob('*.pt')) # list(images_path.glob('*.png'))
        images_list_str = [ str(x) for x in images_list ]
        masks_list = list(Path(gt_path).glob('*.pt')) # list(images_path.glob('*.png'))
        masks_list_str = [ str(x) for x in masks_list ]
        self.images = images_list_str
        self.masks = masks_list_str
        self.images.sort()
        self.masks.sort()
        self.trans = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True)
        ])
        
    def __getitem__(self, index):
        img = self.trans(torch.load(self.images[index]))
        gt = self.trans(torch.load(self.masks[index]))
        gt = torch.where(gt>0.5, 1.0, 0.)
        return img, gt

    def __len__(self):
        return len(self.images)

class SquarePad(object):
    '''
    用于将图像零填充至高度和宽度相同，同时尽可能使原始有意义的图像居中
    思路是，针对给定图像，求出其在四个方向需要填充多少行/列
    较长的边的方向不必填充，较短的边的两个方向上进行尽量相同数量行/列的填充
    '''
    def __init__(self):
        pass

    def __call__(self, img): 
        c, h, w = img.shape
        max_ = max(h,w)
        pad_t = pad_d = h_pad = (max_-h)/2
        if isinstance(h_pad, float):
            pad_t = int(h_pad - .5)
            pad_d = int(h_pad + .5)
        pad_l = pad_r = w_pad = (max_-w)/2
        if isinstance(w_pad, float):
            pad_l = int(w_pad - .5)
            pad_r = int(w_pad + .5)
        img = transforms.functional.pad(img, [pad_l, pad_t, pad_r, pad_d], 0, 'constant')
        return img


class image_augmentor:
    def __init__(self, image_path, mask_path, type, image_size = 256) -> None:
        self.transform_RGB = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 常用标准化，此处暂且不用
            SquarePad(),
            transforms.Resize((image_size, image_size), antialias=True), # 缩放图片(Image)，最短边为image_size
        ])
        self.transform_gray = transforms.Compose([
            transforms.ToTensor(),
            SquarePad(),
            transforms.Resize((image_size, image_size), antialias=True), # 缩放图片(Image)，最短边为image_size
        ])
        self.transform_aug = transforms.Compose([
            transforms.RandomVerticalFlip(), #随机垂直翻转
            transforms.RandomHorizontalFlip(),  #随机水平翻转
            transforms.RandomAffine(degrees=180, translate=(0.2, 0.2)) #随机旋转[-180,180)度，同时随机平移，但平移幅度不超过边长的0.2倍
        ])
        self.image_path = image_path
        self.mask_path = mask_path
        self.type = type
        
    def image_aug(self, simg_path, smask_path, ind): # simg_path : single_img_path
        '''
        simg_path: single image's path
        smask_path: single mask's path
        ind: index
        '''
        img = self.transform_RGB(cv2.imread(simg_path))[[2,1,0],:,:] # 颠倒颜色通道，因为cv2按BGR通道读取图像
        mask = self.transform_gray(cv2.imread(smask_path, cv2.IMREAD_GRAYSCALE))
        for _ in range(5):
            seed = random.randint(0, 2147483647)
            torch.manual_seed(seed)
            torch.save(self.transform_aug(img), os.path.join(self.image_path, "img_"+str(ind).zfill(4)+".pt"))
            torch.manual_seed(seed)
            torch.save(self.transform_aug(mask), os.path.join(self.mask_path, "mask_"+str(ind).zfill(4)+".pt"))
            ind += 1
        return ind
    


def isic_dataset_agmentation(input_path, output_path, type, image_size = 256):
    '''
    此函数用于增强原始ISIC2016数据集
    input_path: ISIC2016数据集的存储位置
    output_path: 增强后的数据集存储根目录
    '''
    # 读取数据
    if type == 'train':
        data_path, gt_path = os.path.join(input_path, "ISBI2016_ISIC_Part1_Train_Data"), os.path.join(input_path, "ISBI2016_ISIC_Part1_Train_GroundTruth")
    elif type == 'test':
        data_path, gt_path = os.path.join(input_path, "ISBI2016_ISIC_Part1_Val_Data"), os.path.join(input_path, "ISBI2016_ISIC_Part1_Val_GroundTruth")
    images_list = list(Path(data_path).glob('*.jpg')) # list(images_path.glob('*.png')) # 并非顺序
    images_list_str = [ str(x) for x in images_list ]
    images = images_list_str
    images.sort()
    masks_list = list(Path(gt_path).glob('*.png')) # list(images_path.glob('*.png')) # 并非顺序
    masks_list_str = [ str(x) for x in masks_list ]
    masks = masks_list_str
    masks.sort()

    # 数据增强并存储，按索引
    index=1
    image_path = os.path.join(output_path, "ISIC_"+type+"_image") # 原始图像存放folder
    mask_path = os.path.join(output_path, "ISIC_"+type+"_groundtruth") # 掩码图像存放folder
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(mask_path):
            os.makedirs(mask_path)
    # 依次对每一张图像增强并存储
    augmentor = image_augmentor(image_path, mask_path, type, image_size)
    for i in range(len(images)):
        index = augmentor.image_aug(images[i], masks[i], index) # index: 每次存储的起始索引



from PIL import Image
import pandas as pd

class ISIC_ori_test_Dataset(Dataset):
    def __init__(self, data_path):
        mode = 'Test'

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        id = name.split('/')[1].split('.')[0]
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        img = self.tensor_transform(img)
        mask = self.tensor_transform(mask)

        return (img, mask, id)
    

if __name__ == '__main__':
     i_path = "E:\dataset\VOCdevkit_All\ISIC_All\Val" # ISIC2016原始数据集根目录
     o_path = "E:\dataset\VOCdevkit_All\SIC_All_augmentation" # 增强数据集预备存储的根目录
     isic_dataset_agmentation(i_path, o_path, "train")
     isic_dataset_agmentation(i_path, o_path, "test")
    
# 测试数据集
# if __name__ == '__main__':
#     path = "E:\DATA\ISIC2016\ISIC_augmentation" # 增强数据集根目录
#     dataset = ISIC_aug_dataset(path, type='train')
#     dataset = ISIC_aug_dataset(path, type='test')