import os
import glob

path_images_root = "/data/visdial/images/"


path_coco_train = os.path.join(path_images_root, "train2014")
path_coco_val = os.path.join(path_images_root, "val2014")
path_visdial_val = os.path.join(path_images_root, "VisualDialog_val2018")
path_visdial_test = os.path.join(path_images_root, "VisualDialog_test2018")


coco_train = glob.glob(os.path.join(path_coco_train, '*'))
coco_val = glob.glob(os.path.join(path_coco_val, '*'))


len(coco_train), len(coco_val), len(coco_train) + len(coco_val)


visdial_val = glob.glob(os.path.join(path_visdial_val, '*'))
visdial_test = glob.glob(os.path.join(path_visdial_test, '*'))
