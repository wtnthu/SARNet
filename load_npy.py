from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pdb


def default_loader(path):
    #pdb.set_trace()
    return np.load(path)

def freq_loader(path):
    #pdb.set_trace()
    return np.load(path)

def gt_loader(path):
    #pdb.set_trace()
    #return Image.open(path).convert('RGB')
    #return Image.open(path).convert('L')
    return np.load(path)



class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader_t=default_loader, loader_f=freq_loader, loader_gt=gt_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],words[1],words[2]))
            
        '''
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                            "Supported extensions are: " + ",".join(extensions)))
        '''
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader_t = loader_t
        self.loader_f = loader_f
        self.loader_gt = loader_gt

    def __getitem__(self, index):
        fn1, fn2, fn3 = self.imgs[index]
        self.img_f = self.loader_f(fn1)
        self.img_t = self.loader_t(fn2)
        self.img_gt = self.loader_gt(fn3)
        #if self.transform is not None:
        self.img_f, self.img_t, self.img_gt= self.transform(self.img_f, self.img_t, self.img_gt, fn1)
        return self.img_f, self.img_t, self.img_gt

    def __len__(self):
        return len(self.imgs)


#train_data=MyDataset(txt='./train.txt', transform=transforms.ToTensor())
#data_loader = DataLoader(train_data, batch_size=100,shuffle=True)
#print(len(data_loader))


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')

'''
for i, (batch_x, batch_y) in enumerate(data_loader):
    if(i<4):
        print(i, batch_x.size(),batch_y.size())
        show_batch(batch_x)
        plt.axis('off')
        plt.show()
'''
