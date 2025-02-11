from pkg_resources import invalid_marker
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
from os import path
import numpy as np
import torch
import glob, os, random
import torchvision.transforms.functional as TF
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, rain_L_dir, rain_H_dir, gt_dir):
        super().__init__()
        self.input_names_L, self.gt_names_L = self.getRainLImageNames(train_data_dir, rain_L_dir, gt_dir)
        self.input_names_H, self.gt_names_H = self.getRainHImageNames(train_data_dir, rain_H_dir, gt_dir)
        self.input_names = self.input_names_L + self.input_names_H
        self.gt_names = self.gt_names_L + self.gt_names_H
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def getRainLImageNames(self, root_dir, image_dir, gt_dir):
        input_dir = os.path.join(root_dir, image_dir)
        output_dir = os.path.join(root_dir, gt_dir)
        image_names_tmp = []
        image_names = []
        gt_names = []
        for file in os.listdir(input_dir):
            if file.endswith(".png"):
                in_name = os.path.join(input_dir, file)
                image_names_tmp.append(in_name)
        for in_name in image_names_tmp:
            gt_name = in_name.replace(image_dir, gt_dir).replace("Rain_L_", "No_Rain_")
            if os.path.exists(gt_name):
                image_names.append(in_name)
                gt_names.append(gt_name)
        return image_names, gt_names

    def getRainHImageNames(self, root_dir, image_dir, gt_dir):
        input_dir = os.path.join(root_dir, image_dir)
        output_dir = os.path.join(root_dir, gt_dir)
        image_names_tmp = []
        image_names = []
        gt_names = []
        for file in os.listdir(input_dir):
            if file.endswith(".png"):
                in_name = os.path.join(input_dir, file)
                image_names_tmp.append(in_name)
        for in_name in image_names_tmp:
            gt_name = in_name.replace(image_dir, gt_dir).replace("Rain_H_", "No_Rain_")
            if os.path.exists(gt_name):
                image_names.append(in_name)
                gt_names.append(gt_name)
        return image_names, gt_names
    
    def preprocess(self, image):
        image = cv2.resize(image, (640, 480))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        return image
    
    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        input_image = cv2.imread(input_name)
        gt_image = cv2.imread(gt_name)

        input_im = self.preprocess(input_image)
        gt = self.preprocess(gt_image)
        return input_im, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

class TrainData_new(data.Dataset):
    def __init__(self, crop_size, train_data_dir,train_filename):
        super().__init__()
        train_list = train_data_dir + train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input','gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/',input_name)[-1][:-4]
        
        input_img = Image.open(self.train_data_dir + input_name)


        try:
            gt_img = Image.open(self.train_data_dir + gt_name)
        except:
            gt_img = Image.open(self.train_data_dir + gt_name).convert('RGB')

        width, height = input_img.size
        tmp_ch = 0

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size
        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])

        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        
        # --- Check the channel is 3 or not --- #
        # print(input_im.shape)
        if list(input_im.shape)[0] != 3 or list(gt.shape)[0] != 3:
            raise Exception('Bad image channel: {}'.format(gt_name))


        return input_im, gt, img_id,R_map,trans_map

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)