import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

MIN_INPUT_VALUE = 0
MAX_INPUT_VALUE = 1
WINDOWING_MIN_MAX = np.array(((-20, 100), (-20, 25), (25, 60)))
INPUT_SIZE = OUTPUT_SIZE = MASK_SIZE = (224, 224)
INPUT_SHAPE = (*INPUT_SIZE, len(WINDOWING_MIN_MAX))
EMPTY_CANVAS = np.zeros((*INPUT_SIZE, 3), dtype="uint8") + 255 # 3 color channels, white colors
OUTPUT_CHANNELS = 3 
MAP_COLORS = np.array(
    [
        [  0,   0,   0], # black
        # [215, 175, 168], # brownish pink # suzuhara lulu
        [ 56, 180, 139],  # green # forgot who is this
        [ 50,  76, 172], # dark blue # aiba uiha 
        # [ 66, 255, 255], # cyan  # forgot this one as well
     
    ], dtype="uint8")

def load_nifti(nifti_path, dtype="int16"): #load nifti and create array
    data_nifti = nib.load(str(nifti_path))
    data_array = data_nifti.get_fdata().astype(dtype)
    return data_array, data_nifti


def window_ct_v1(ct_scan, w_level, w_width): #make windowing
    ct_scan = ct_scan.copy()
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    num_slices = ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:,:,s]
        slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
        slice_s[slice_s < 0] = 0
        slice_s[slice_s > 255] = 255
        ct_scan[:,:,s] = slice_s
    return ct_scan

# didn't test yet
def window_ct_v2(ct_scan, w_level, w_width):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    ct_scan = (ct_scan - w_min) * (255 / (w_max-w_min))
    ct_scan[ct_scan < 0] = 0
    ct_scan[ct_scan > 255] = 255
    ct_scan = ct_scan.astype("int16")    
    return ct_scan

def window_ct_v3(data_array, w_min, w_max, min_value=0, max_value=255, dtype="uint8"):
    data_array = ((data_array - w_min) / (w_max - w_min) * (max_value - min_value)) + min_value
    data_array = np.where(data_array < min_value, min_value, data_array) #if < min -> min else data_array
    data_array = np.where(data_array > max_value, max_value, data_array)
    return data_array.astype(dtype)

def squishes(data_array): #find min max to norm
    min_v = np.min(data_array)
    data_array = data_array - min_v
    data_array = data_array / np.max(data_array)
    data_array = data_array * 255
    return data_array.astype("uint8")

def create_dir_if_not_exist(*paths: Path):
    for p in paths:
        p.mkdir(exist_ok=True)

def print_param_list(*args):
    print("======Parameter======")
    for i in args:
        print(i)
    print("=====================")

def timer_wrapper(f):
    def f_timer(*args, **kwargs):
        from time import time
        t = time()
        o = f(*args, **kwargs)
        t = time() - t
        return o, t
    return f_timer

def tester_wrapper(f):
    def f_tester(*args, **kwargs):
        print("test both these func will work the same?")
        print("test might take around 40 sec")
        f(*args, **kwargs)
        print("both func work the same")
    return f_tester

def create_mask(one_hot_mask):
    """
    (width, height, n_classes) -> (width, height, 1)
    (batch, width, height, n_classes) -> (batch, width, height, 1)
    """
    pred_mask = np.argmax(one_hot_mask, axis=-1) 
    return pred_mask[..., np.newaxis]


def create_visible_mask(gt, mask, is_one_hot=False):
    if is_one_hot:
        mask = create_mask(mask)
    if np.max(gt) <= 1.0: # for floating-point `gt`, looking for better way to do this
        gt = gt * 255
    gt = gt.astype("int16")
    for i in range(OUTPUT_CHANNELS):
        gt = np.where(mask == i, gt + MAP_COLORS[i], gt)
    gt = np.where(gt > 255, 255, gt)
    return gt

COLOR_DIFF = np.array([
    [236, 106, 118],  # red # bun's # for gt diff  
    [ 80, 120,  67]  # green # tin's # for model predicted diff
], dtype="uint8")
def create_diff_mask(img, label, pred_label, is_one_hot=False):
    if is_one_hot:
        label = create_mask(label)
        pred_label = create_mask(pred_label)
    try:
        img = img.numpy()
    except Exception:
        pass
    img = (img * 255).astype("int16")
    csf_gt = np.where(label == 1, 1, 0) # looking for class csf
    csf_pred = np.where(pred_label == 1, 1, 0)
    diff = csf_gt - csf_pred
    
    a = np.where(diff > 0, img + COLOR_DIFF[0], img)
    a = np.where(diff < 0, img + COLOR_DIFF[1], a)
    
    a = np.where((a > 255) & (diff > 0), COLOR_DIFF[0], a)
    a = np.where((a > 255) & (diff < 0), COLOR_DIFF[1], a)

    return a

def plot_mask_side_by_side_v2(img, label=None, predicted_label=None, save_name=None, show=True, figsize=15, **kwargs):
    row = 2
    c = 1
    fig = plt.figure(figsize=(int(figsize*INPUT_SHAPE[2]/row), figsize))
    img = window_ct_v3(img, np.min(img), np.max(img), 0, 1, "float32")
    img_0 = None

    if label is not None:
        # label = create_mask(label)
        label = label[..., np.newaxis]
    if predicted_label is not None:
        # predicted_label = create_mask(predicted_label)
        predicted_label = predicted_label[..., np.newaxis]

    for i in range(INPUT_SHAPE[2]):
        ax1 = fig.add_subplot(row, INPUT_SHAPE[2], c)
        ax1.set_title("gt%d w_min=%d, w_max=%d" % (i, WINDOWING_MIN_MAX[i, 0], WINDOWING_MIN_MAX[i, 1]))
        ax1.imshow(img[..., i], cmap="gray")
        ax1.axis("off")
        c = c + 1
    
    ax4 = fig.add_subplot(row, INPUT_SHAPE[2], c)
    ax4.set_title("true mask on gt0")
    ax4.axis("off")
    if label is not None:
        img_0 = img[..., 0, np.newaxis]
        visible_mask = create_visible_mask(img_0, label, is_one_hot=False)
        ax4.imshow(visible_mask)
    
    else:
        ax4.imshow(EMPTY_CANVAS)
    c = c + 1
    
    ax7 = fig.add_subplot(row, INPUT_SHAPE[2], c)
    ax7.set_title("predicted mask on gt0")
    ax7.axis("off")
    if not predicted_label is None:
        if img_0 is None:
            img_0 = img[..., 0, np.newaxis]
        visible_mask = create_visible_mask(img_0, predicted_label, is_one_hot=False)    
        ax7.imshow(visible_mask)
        c = c + 1
        ax7 = fig.add_subplot(row, INPUT_SHAPE[2], c)
        ax7.axis("off")
        ax7.set_title("diff mask on gt0")
        visible_mask = create_diff_mask(img_0, label, predicted_label, is_one_hot=False)
        ax7.imshow(visible_mask)

    else:
        ax7.imshow(EMPTY_CANVAS)
    c = c + 1
    
    if save_name:
        fig.savefig(str(save_name))
    if show:
        plt.show()
        print()
    fig.clf()
    plt.close(fig)

def create_each_windowing_silice(gt_slice):
    """
    gt_slice: 2d
    """
    gt_slice = cv2.resize(gt_slice, (INPUT_SIZE))
    new_image = np.ones(INPUT_SHAPE, dtype="float32")
    for j in range(INPUT_SHAPE[2]):
        new_image[..., j] = window_ct_v3(gt_slice, 
                                        WINDOWING_MIN_MAX[j, 0], 
                                        WINDOWING_MIN_MAX[j, 1],
                                        min_value=MIN_INPUT_VALUE,
                                        max_value=MAX_INPUT_VALUE, 
                                        dtype="float32")
    return new_image
    
def load_nifti_skip_none(file):
    if file:
        return load_nifti(file)
    return None, None

def load_multi_nifti(file, *args):
    # args = list(filter(lambda x: x is not None, args))
    return list(map(lambda x:load_nifti_skip_none(str(x/file)), args))



