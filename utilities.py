import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from torch import Tensor
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
        else :
            hull = fusion_img
    return hull

def vs_generate(input_mask, gen_mask, fusion):
    err_space = np.float64(np.logical_xor(input_mask, gen_mask))
    corr_space = np.logical_and(input_mask, gen_mask)
    R,G,B = cv2.split(err_space)
    R[R==0] = 0.18
    G[G==0] = 0.46
    G[G>0.47] = 0
    B[B==0] = 0.71
    B[B>0.72] = 0
    err_space =cv2.merge([R,G,B])
    
    err_space *= np.float64(np.logical_not(corr_space))
    # print("err_space",err_space)
    corr_space = np.float64(corr_space)
    corr_space *= fusion
    err_space += corr_space
    return err_space

def compress_channel(input_batch_img, threshold):
    single_img = torch.zeros(1, input_batch_img.size(2), input_batch_img.size(3))
    output_batch_img = torch.zeros(input_batch_img.size(0), 1,
                                   input_batch_img.size(2), input_batch_img.size(3))
    for idx,n in enumerate(input_batch_img):
        for ch,img in enumerate(n):
            single_img[0][ img > threshold ] = ch
        output_batch_img[idx] = single_img
    return output_batch_img

def show_images(input_imgs, input_masks: None, gen_masks= None,
                nrow=5, ncol=1, show: bool = True, save: bool = False, path ="", mode: bool =False):
    # compare and show n*m images from generator in one figure and optionally save it
    if input_imgs.shape[0] < nrow:
        nrow = input_imgs.shape[0]
    figsize = (nrow*3+2,9)
    count = 311
    img_label = ["input\nimages", "input\nmask"]
    inputs = [input_imgs, input_masks]
    offset = -0.1

    if mode == True and gen_masks == None:
        print("Input ERROR! Expected [gen_mask] but got [None].")
        return None
        
    elif mode == True and gen_masks != None:
        figsize = (nrow*3+2,18)
        count = 611
        img_label.append("generated\nmask")
        inputs.append(gen_masks)

    plt.figure(figsize=figsize)
    for imgs, label in zip([imgs for imgs in inputs if input_masks is not None], img_label):
        imgs = imgs[:nrow * ncol]
        imgs = imgs.view(imgs.size(0), imgs.size(1), 800, 800)
        ax = plt.subplot(count)
        ax.set_title(label ,x=offset, y=0.35)
        img = np.transpose(make_grid(imgs, nrow=nrow, padding=2, normalize=True).cpu(), (1, 2, 0))
        img = np.float32(np.array(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img.shape[1]*2, int(img.shape[0]*1.5)), interpolation=cv2.INTER_AREA)
        plt.axis("off")
        plt.imshow(img)

        if label == "input\nmask":
            input_mask = img
        if label == "generated\nmask":
            gen_mask = img

        if label == "input\nimages":
            origin_img = img/3+0.6
        else :
            count+=1
            ax = plt.subplot(count)
            name = label.split("\n")[0] + "\nfusion"
            ax.set_title(name,x=offset, y=0.35)
            fusion = origin_img.copy()
            contours_generate(img,fusion)
            plt.axis("off")
            plt.imshow(fusion)
        
        if label == "generated\nmask":
            count+=1
            ax = plt.subplot(count)
            name = "ground truth\nvs\ngenerated"
            ax.set_title(name,x=offset, y=0.35)
            fusion = origin_img.copy()
            vs = vs_generate(input_mask, gen_mask, fusion)
            #print(vs,)
            plt.axis("off")
            plt.imshow(vs)
        count+=1
    if save:
        plt.savefig(path)
    if show:
        plt.show()

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def pixelAccuracy(gen_mask, input_mask):
    pixel_labeled = torch.sum(input_mask > 0).float()
    pixel_corr = torch.sum((gen_mask == input_mask) * (input_mask > 0)).float()
    pixel_acc = pixel_corr / (pixel_labeled + 1e-10)

    return pixel_acc, pixel_corr, pixel_labeled

def MeanPixelAccuracy(gen_mask, input_mask):    
    pixel_acc = np.empty(input_mask.shape[0])
    pixel_corr = np.empty(input_mask.shape[0])
    pixel_labeled = np.empty(input_mask.shape[0])

    for i in range(input_mask.shape[0]):
        pixel_acc[i], pixel_corr[i], pixel_labeled[i] = \
        pixelAccuracy(gen_mask[i], input_mask[i])

    acc = 100.0 * np.sum(pixel_corr) / (np.spacing(1) + np.sum(pixel_labeled))

    return acc

def intersectionAndUnion(gen_mask, input_mask, numClass=8):
    gen_mask = gen_mask * (input_mask > 0).long()
    intersection = gen_mask * (gen_mask == input_mask).long()
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    (area_pred, _) = np.histogram(gen_mask, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(input_mask, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    IoU = area_intersection / (area_union + 1e-10)

    return IoU, area_intersection, area_union

def mIoU(gen_mask, input_mask):
    area_intersection = []
    area_union = []

    for i in range(input_mask.shape[0]):
        _, intersection, union = intersectionAndUnion(gen_mask[i], input_mask[i])
        area_intersection.append(intersection)
        area_union.append(union)

    IoU = 1.0 * np.sum(area_intersection, axis=0) / np.sum(np.spacing(1)+area_union, axis=0)

    return np.mean(IoU)

EPS = 1e-10
def nanmean(x):
    return torch.mean(x[x == x])

def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def per_class_pixel_accuracy(hist):
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc, per_class_acc

def per_class_PA (pred, label):
    hist = _fast_hist(pred, label,8)
    avg, per = per_class_pixel_accuracy(hist)
    return avg, per

def all_metrics(gen_mask, input_mask):
    acc,cor,lab = pixelAccuracy(gen_mask, input_mask)
    acc = acc.item()
    m_acc = MeanPixelAccuracy(gen_mask, input_mask)
    IOU,_,_ = intersectionAndUnion(gen_mask, input_mask)
    MIOU = mIoU(gen_mask, input_mask)
    avg, per = per_class_PA(gen_mask, input_mask)
    DICE = multiclass_dice_coeff(gen_mask, input_mask)
    DICE = DICE.item()
    print(f"PA:{acc}, mPA:{m_acc}, Dice:{DICE}, IoU:{IOU}, mIoU:{MIOU}")
