from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import scipy.io as sio

class VOC2012Segmentation(Dataset):
    """ PyTorch Dataset of VOC 2012"""
    def __init__(self, root_dir, train=True, transform=None, size = 224):
        """
        :param root_dir: root_dir for VODdevkit
        :param transform:
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.file_list = self._get_file_list()
        self.mask_transform  = transforms.Compose([transforms.Resize((size,size)),
                                                    transforms.ToTensor()])

    def _get_file_list(self):
        txt_path = os.path.join(self.root_dir, "VOC2012/ImageSets/Segmentation/{}.txt".format("train" if self.train else "val"))
        with open(txt_path,"r") as f:
            file_list = [k.strip() for k in f.readlines()]
        return file_list

    def __getitem__(self,idx):
        img_root = os.path.join(self.root_dir,"VOC2012/JPEGImages/")
        img_name = self.file_list[idx]+".jpg"
        img_path = os.path.join(img_root,img_name)
        img = Image.open(img_path)

        mask_root = os.path.join(self.root_dir,"VOC2012/SegmentationClass/")
        mask_name = self.file_list[idx]+".png"
        mask_path = os.path.join(mask_root,mask_name)
        mask = Image.open(mask_path)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)*255

        return img, mask.long()

    def __len__(self):
        return len(self.file_list)

class SBD(Dataset):
    """ http://home.bharathh.info/pubs/codes/SBD/download.html """
    def __init__(self, root_dir, train=True, transform=None, size = 224):
        """
        :param root_dir: root_dir for VODdevkit
        :param transform:
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.file_list = self._get_file_list()
        self.mask_transform  = transforms.Compose([transforms.Resize((size,size)),
                                                    transforms.ToTensor()])

    def _get_file_list(self):
        txt_path = os.path.join(self.root_dir, "dataset/{}.txt".format("train" if self.train else "val_no_cbd"))
        with open(txt_path,"r") as f:
            file_list = [k.strip() for k in f.readlines()]
        return file_list

    def __getitem__(self,idx):
        img_root = os.path.join(self.root_dir,"dataset/img/")
        img_name = self.file_list[idx]+".jpg"
        img_path = os.path.join(img_root,img_name)
        img = Image.open(img_path)

        mask_root = os.path.join(self.root_dir,"dataset/cls/")
        mask_name = self.file_list[idx]+".mat"
        mask_path = os.path.join(mask_root,mask_name)
        mat = sio.loadmat(mask_path)
        mask_array = mat['GTcls'][0][0][1]
        mask = Image.fromarray(mask_array)
        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)*255

        return img, mask.long()

    def __len__(self):
        return len(self.file_list)

class VOC2012_without_CBD(Dataset):
    """ http://home.bharathh.info/pubs/codes/SBD/download.html """
    def __init__(self, root_dir, train=True, transform=None, size = 224):
        """
        :param root_dir: root_dir for VODdevkit
        :param transform:
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.file_list = self._get_file_list()
        self.mask_transform  = transforms.Compose([transforms.Resize((size,size)),
                                                    transforms.ToTensor()])

    def _get_file_list(self):
        txt_path = "./val_no_cbd.txt"
        with open(txt_path,"r") as f:
            file_list = [k.strip() for k in f.readlines()]
        return file_list

    def __getitem__(self,idx):
        img_root = os.path.join(self.root_dir,"dataset/img/")
        img_name = self.file_list[idx]+".jpg"
        img_path = os.path.join(img_root,img_name)
        img = Image.open(img_path)

        mask_root = os.path.join(self.root_dir,"dataset/cls/")
        mask_name = self.file_list[idx]+".mat"
        mask_path = os.path.join(mask_root,mask_name)
        mat = sio.loadmat(mask_path)
        mask_array = mat['GTcls'][0][0][1]
        mask = Image.fromarray(mask_array)
        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)*255

        return img, mask.long()

    def __len__(self):
        return len(self.file_list)



def transform(config):
    transform = transforms.Compose([transforms.Resize((config.img_size,config.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
    return transform

def get_loader(config,train=True):
    assert os.path.exists(config.data_path)
    if config.data.lower() == 'voc2012':
    	dset = VOC2012Segmentation(root_dir=config.data_path,train=train, transform = transform(config), size = config.img_size)
    elif config.data.lower() == 'cbd':
        dset = SBD(root_dir=config.data_path, train=train, transform = transform(config), size = config.img_size)
    elif config.data.lower() == 'hybrid':
        if train:
            dset = SBD(root_dir=config.data_path, train=train, transform=transform(config), size=config.img_size)
        else:
            dset = VOC2012Segmentation(root_dir=config.val_data_path, train=train, transform=transform(config),
                                       size=config.img_size)


    loader = DataLoader(dset, batch_size = config.batch_size, shuffle = True if train else False)
    return loader


