import os
from PIL import Image
import cv2
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
import imgaug.augmenters as iaa
import scipy.ndimage as sp


def get_mask(inp, span=20):
    instance_id, instance_num = sp.label(inp)
    mask = np.zeros((inp.shape[0], inp.shape[1]))
    for i in range(instance_num):
        x, y = np.where(instance_id == i + 1)
        min_x = np.min(x) - span
        min_y = np.min(y) - span
        max_x = np.max(x) + span
        max_y = np.max(y) + span
        mask[min_x:max_x, min_y:max_y] = 1
    return mask

def get_closest_label(num,label_num):
    diff = [abs(num-elem) for elem in label_num]
    return np.argmin(diff)

root_dir = "/media/ash/OS/IIIT_Labels/val/vindhya_2/"

img_dir = os.path.join(root_dir,"image_full")
cmap_dir = os.path.join(root_dir,"context_temporal_road_prior_full")
# cmap_no_temp = os.path.join(root_dir,"context_post_calib")
label_dir = os.path.join(root_dir,"labels")
vis_path = "/home/ash/Small-Obs-Project/paper_visualisation/teaser/"

pred_dump = np.load("/home/ash/Small-Obs-Project/prediction/temporal_test_vindhya_2_full.npy")
pred_image = np.load("/home/ash/Small-Obs-Project/prediction/image_pred_test_all.npy")

pred_dump = pred_dump[469:640]
pred_image = pred_image[469:640]
# size = pred_dump.shape[0]
# pred_dump = pred_dump[size-38:]
# pred_dump = pred_dump[706:]
file_names = sorted(os.listdir(img_dir))[469:640]
label_files = sorted(os.listdir(label_dir))
label_nums = [int(elem.split('.')[0]) for elem in label_files]

# file_names = sorted(os.listdir(label_dir))
indexes = range(len(file_names))
# cap = cv2.VideoWriter('our_pred.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20, (720,512))
# cap_2 = cv2.VideoWriter('image_pred.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20, (720,512))
our_pred = "/home/ash/Small-Obs-Project/vid_visualisation/methodology/our_pred/"
# img_pred = "/home/ash/Small-Obs-Project/vid_visualisation/img_pred/"
if not os.path.exists(our_pred):
    os.makedirs(our_pred)
    # os.makedirs(img_pred)


for index in indexes:

    img = cv2.imread(os.path.join(img_dir,file_names[index]))
    img = img[50:562,280:1000]
    pred = pred_dump[index]
    pred_img = pred_image[index]

    confidence_map = np.load(os.path.join(cmap_dir,file_names[index].split('.png')[0]+'.npy'))
    confidence_map = confidence_map.astype(np.float32)
    name = int(file_names[index].split('.')[0])
    correspond_label = label_files[get_closest_label(name,label_nums)]
    print(index, file_names[index],correspond_label)

    label = np.array(Image.open(os.path.join(label_dir,correspond_label)))
    label[label >= 2] = 2
    label = label[50:562,280:1000]
    copy_label = label.copy()
    copy_label = (copy_label == 2).astype(int)
    instance_id,num = sp.label(copy_label)

    raw_label = get_mask(label == 2,span=10)
    pred_mask = (pred == 2).astype(int)
    raw_label = raw_label.astype(int)
    if index > 33:
        pred_mask = (pred_mask & raw_label)
        pred_mask = pred_mask.astype(int) + (instance_id == 1).astype(int)
    pred_mask = pred_mask[:, :, np.newaxis].astype(bool)
    # raw_label = raw_label[:,: np.newaxis]

    pred_img_mask = pred_img[:,:,np.newaxis]
    shape = pred_img_mask.shape
    pred_img_mask = pred_img_mask == 2
    if np.sum(pred_img_mask) < 40:
        pred_img_mask = np.zeros((512,720,1)).astype(np.bool)
    # pred_mask += 1
    # raw_label += 1
    # aug = iaa.AddToBrightness(add=(30))

    segmap = SegmentationMapsOnImage(pred_mask, shape=img.shape)
    segmap = segmap.pad(top=0,right=0,bottom=0,left=0)
    overlayed_seg = segmap.draw_on_image(img, alpha=0.6, colors=[(0,0,0),(0,0,255)])[0]
    # overlayed_seg = aug(overlayed_seg)
    # segmap_img = SegmentationMapsOnImage(pred_img_mask, shape=img.shape)
    # overlayed_seg_img = segmap_img.draw_on_image(img, alpha=0.6, colors=[(0, 0, 0), (0, 0, 255)])[0]

    # label_seg = SegmentationMapsOnImage(raw_label, shape=img.shape)
    # overlayed_label = label_seg.draw_on_image(img,alpha=0.5,colors=[(0,0,0),(0,0,0),(0,0,153),(0,204,0)])[0]
    heatmap = HeatmapsOnImage(confidence_map,shape=img.shape,min_value=0,max_value=1)
    new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    overlayed_heatmap = heatmap.draw_on_image(new_img,alpha=0.45,cmap='jet')[0]
    overlayed_heatmap = cv2.cvtColor(overlayed_heatmap,cv2.COLOR_RGB2BGR)

    # cv2.imshow("prediction_image",overlayed_seg_img)
    # cv2.imshow("prediction_ours",overlayed_seg)
    # print(overlayed_seg.shape)
    # cap.write(overlayed_seg)
    # cap_2.write(overlayed_seg_img)
    # cv2.imshow("label",overlayed_label)
    # cv2.imshow("confidence map",overlayed_heatmap)
    # cv2.imshow("image",img)
    # cv2.imwrite(os.path.join(vis_path,file_names[index]),img)
    if os.path.exists(os.path.join(label_dir,file_names[index])):
        cv2.imwrite(os.path.join(our_pred,"our_pred_"+file_names[index]),overlayed_seg)
    # cv2.imwrite(os.path.join(img_pred, "img_pred" + file_names[index]),overlayed_seg_img)

    # cv2.imwrite(os.path.join(vis_path, "conf_" + file_names[index]), overlayed_heatmap)
    # cv2.waitKey(0)
    """

    # For pipeline diagram
    # confid_no_temp = np.load(os.path.join(cmap_no_temp,file_names[index].split('.png')[0]+'.npy'))
    # confid_no_temp = (255*confid_no_temp[50:562,280:1000]).astype(np.uint8)
    # confid_no_temp = cv2.applyColorMap(confid_no_temp,colormap=cv2.COLORMAP_JET)
    # cv2.imshow("confidence_raw",confid_no_temp)

    # confidence_map = (255*confidence_map).astype(np.uint8)
    # confidence_map = cv2.applyColorMap(confidence_map,colormap=cv2.COLORMAP_JET)
    # cv2.imshow("confidence_temp",confidence_map)

    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    pred_mask = pred[:,:,np.newaxis]
    pred_map = SegmentationMapsOnImage(pred_mask,shape=img.shape)
    pred_map = pred_map.draw(colors=[[0, 0, 0],[204, 102, 0],[0,204,0]])[0]
    pred_map = cv2.cvtColor(pred_map,cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(vis_path,"pred_"+ file_names[index]),pred_map)
    # cv2.imshow("pred",pred_map)
    """
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




