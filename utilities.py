import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
# functions to show an image
def contours_generate(input_img,fusion_img):
    input_img = np.float32(input_img)
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    img_gray *= 255
    colors = [(255,50,0),(131,60,11),(0,255,0),(0,0,255),(255,0,255),(255,0,0),(0,0,128)]
    for threshhold in range(1,8):
        ret, thresh = cv2.threshold(np.uint8(img_gray),(threshhold*36-1), 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 3, 2)
        if contours:
            if threshhold == 1:
                hull = cv2.drawContours(fusion_img, contours, -1, colors[threshhold-2], 6)
            else:
                hull = cv2.drawContours(hull, contours, -1, colors[threshhold-2], 6)
    return hull

def vs_generate(input_mask, gen_mask, fusion):
    err_space = np.float64(np.logical_xor(input_mask, gen_mask))
    corr_space = np.logical_and(input_mask, gen_mask)
    
    R,G,B = cv2.split(err_space)
    R[R==0] = 0.18
    G[G==0] = 0.46
    G[G>0.46] = 0
    B[B==0] = 0.71
    B[B>0.71] = 0
    err_space =cv2.merge([R,G,B])
    err_space *= np.float64(np.logical_not(corr_space))
    corr_space = np.float64(corr_space)
    corr_space *= fusion
    err_space += corr_space
    return err_space

def show_images(input_imgs, input_masks: None, gen_masks= None,
                nrow=5, ncol=1, show: bool = True, name="", channels=3, mode: bool =False):
    # compare and show n*m images from generator in one figure and optionally save it
    figsize = (11,6)
    count = 311
    img_label = ["input images", "input mask"]
    inputs = [input_imgs, input_masks]

    if mode == True and gen_masks == None:
        print("Input ERROR! Expected [gen_mask] but got [None].")
        return None
        
    elif mode == True and gen_masks != None:
        figsize = (11,10)
        count = 611
        img_label.append("generated mask")
        inputs.append(gen_masks)

    plt.figure(figsize=figsize)
    for imgs, label in zip([imgs for imgs in inputs if input_masks is not None], img_label):
        imgs = imgs[:nrow * ncol]
        imgs = imgs.view(imgs.size(0), imgs.size(1), 800, 800)
        ax = plt.subplot(count)
        ax.set_title(label ,x=-0.1, y=0.4)
        img = np.transpose(make_grid(imgs, nrow=nrow, padding=2, normalize=True).cpu(), (1, 2, 0))
        img = np.float32(np.array(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img.shape[1]*2, int(img.shape[0]*1.5)), interpolation=cv2.INTER_AREA)
        plt.axis("off")
        plt.imshow(img)

        if label == "input mask":
            input_mask = img
        if label == "generated mask":
            gen_mask = img

        if label == "input images":
            origin_img = img/3+0.6
        else :
            count+=1
            ax = plt.subplot(count)
            name = label.split(" ")[0] + " fusion"
            ax.set_title(name,x=-0.1, y=0.4)
            fusion = origin_img.copy()
            hull = contours_generate(img,fusion)
            plt.axis("off")
            plt.imshow(fusion)
        
        if label == "generated mask":
            count+=1
            ax = plt.subplot(count)
            name = "gt vs pred"
            ax.set_title(name,x=-0.1, y=0.4)
            fusion = origin_img.copy()
            vs = vs_generate(input_mask, gen_mask, fusion)
            #print(vs,)
            plt.axis("off")
            plt.imshow(vs)
        count+=1

    if show:
        plt.show()
