
from PIL import Image
import imgviz
import cv2
import argparse
import os
import numpy as np
from numpy.core.defchararray import asarray
import tqdm
import re






def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img


def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)

    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    img_main = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
    return img_main



def rescale_src(mask_src, img_src, h, w):
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w), dtype=np.uint8)
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


def Large_Scale_Jittering(mask, img, min_scale=0.1, max_scale=2.0):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w), dtype=np.uint8)
        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_pad[y:y+h_new, x:x+w_new] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
        mask_crop = mask[y:y+h, x:x+w]
        return mask_crop, img_crop


def copy_paste(mask_src, img_src, mask_main, img_main):
    mask_src, img_src = random_flip_horizontal(mask_src, img_src)
    mask_main, img_main = random_flip_horizontal(mask_main, img_main)

    # LSJ， Large_Scale_Jittering
    lsj=True
    if lsj:
        mask_src, img_src = Large_Scale_Jittering(mask_src, img_src)
        mask_main, img_main = Large_Scale_Jittering(mask_main, img_main)
    else:
        # rescale mask_src/img_src to less than mask_main/img_main's size
        h, w, _ = img_main.shape
        mask_src, img_src = rescale_src(mask_src, img_src, h, w)

    img = img_add(img_src, img_main, mask_src)
    mask = img_add(mask_src, mask_main, mask_src)

    return mask, img







def main(args):
    #directories
    main_img_dir=args.input_dir
    main_mask_dir=os.path.join(main_img_dir,"..","annotation")
    out_dir=args.output_dir
    source_dir=args.source_dir

    out_train=os.path.join(out_dir,"train")
    out_dev=os.path.join(out_dir,"dev")
    train_img=os.path.join(out_train,"images")
    train_mask=os.path.join(out_train,"masks")
    dev_img=os.path.join(out_dev,"images")
    dev_mask=os.path.join(out_dev,"masks")



    os.makedirs(train_img,exist_ok=True)
    os.makedirs(train_mask,exist_ok=True)  #create folders
    os.makedirs(dev_img,exist_ok=True)
    os.makedirs(dev_mask,exist_ok=True)


    #get main image names
    main_img_name=os.listdir(main_img_dir)
    main_mask_name=os.listdir(main_mask_dir)

    #get source image names
    source_img_name=os.listdir(source_dir)



    
    index=0

    #iterates them
    for img in main_img_name:
        name=img.split('.')[0] #get name
        img_dir=os.path.join(main_img_dir,img) #directory of main image
        mask_dir=os.path.join(main_img_dir,'..','annotation',name+'.png') # directory of main mask

        print("Pocessing "+name)

        tbar=tqdm.tqdm(range(args.nums),ncols=100)
        #generate samples
        for num in tbar:
            #choose a source image randomly
            source_img=np.random.choice(source_img_name)
            source_img_dir=os.path.join(source_dir,source_img)
            #directory
            img_src=cv2.imread(source_img_dir)
            mask_src=np.asarray(np.zeros((img_src.shape[0],img_src.shape[1])),dtype=np.uint8)
            img_main=cv2.imread(img_dir)
            mask_main=np.asarray(Image.open(mask_dir),dtype=np.uint8)   
            
            #generate!
            mask, img = copy_paste(mask_main, img_main,mask_src, img_src)



            if((index%300)<75): #save as dev set
                cv2.imwrite(os.path.join(dev_img,str(index)+'.jpg'),img)
                cv2.imwrite(os.path.join(dev_mask,str(index)+'.png'),mask)
            else:  #save as training set
                cv2.imwrite(os.path.join(train_img,str(index)+'.jpg'),img)
                cv2.imwrite(os.path.join(train_mask,str(index)+'.png'),mask)
            index=index+1

            #save


        









def get_args():
    parser=argparse.ArgumentParser(description="Copy-Paste Pocess")
    parser.add_argument("-i","--input_dir",type=str,help="directory of input images",default="./gt_resize/support/images")
    parser.add_argument("-o","--output_dir",type=str,help="directory that you want to store your results",default=\
        "./generations")
    parser.add_argument('-s',"--source_dir",type=str,help="directory of source images",default="./train2017/")
    parser.add_argument('-n',"--nums",type=int,help="number of datas for each foreground",default=300)
    args=parser.parse_args()

    return args





if __name__=="__main__":
    args=get_args()
    main(args)
