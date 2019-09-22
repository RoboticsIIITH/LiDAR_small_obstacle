import lidr_clustering_orig as lpc
import cv2
import numpy as np
from PIL import Image
import os

wk = 33
path = '/media/ash/OS/small_obstacle_bag/synced_data/seq_2/'
image_path = os.path.join(path, 'image')
label_path = os.path.join(path, 'labels')
depth_path = os.path.join(path, 'depth')
ring_path = os.path.join(path, 'rings')
images = [os.path.join(image_path, i) for i in sorted(os.listdir(label_path))]
labels = [os.path.join(label_path, i) for i in sorted(os.listdir(label_path))]
depths = [os.path.join(depth_path, i) for i in sorted(os.listdir(label_path))]
rings = [os.path.join(ring_path, i) for i in sorted(os.listdir(label_path))]


# OUR CHOICE OF FEATURE EXTRACTORS
# sift
# surf
# ORB
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

def create_mask(keypts,img_shape,y_feat,x_feat,index,span=15):
    mask = np.zeros((img_shape[0],img_shape[1]))
    for point in keypts:
        if point is not None:
            x, y = int(point.pt[0]), int(point.pt[1])
            if x<1280-span and x>span and y<720-span and y>span:
                mask[y - span: y + span, x - span:x + span] = 1

    lid_pts = np.zeros((img_shape[0], img_shape[1]))
    lid_pts[x_feat[index],y_feat[index]] = 1
    lid_pts = lid_pts*mask
    cnt = []
    hull = []
    for i in range(lid_pts.shape[0]):
        for j in range(lid_pts.shape[1]):
            if lid_pts[i,j]:
                cnt.append([i,j])
    if len(cnt):
        cnt = np.array(cnt)
        hull = cv2.convexHull(cnt,returnPoints=True)
    return mask,hull


for i, image in enumerate(images):
    path = {}
    path['image'] = images[i]
    path['label'] = labels[i]
    path['ring'] = rings[i]
    path['depth'] = depths[i]
    x_features, y_features, cluster_preds, value_features = lpc.main(draw=False, path=True, paths=path)
    if i == 0:
        continue
    mask_template = (np.asarray(Image.open(labels[i]), dtype=np.uint8) == 2)
    mask_template = np.asarray(mask_template, dtype=np.uint8)
    # mask_reference = (np.asarray(Image.open(labels[i-1]), dtype=np.uint8) == 2)
    # mask_reference = np.asarray(mask_reference, dtype=np.uint8)
    # temp_uni = np.unique(mask_template)
    #ref_uni = np.unique(mask_reference)
    # if len(temp_uni) == 1 or len(ref_uni) == 1:
    #     continue
    img_template = cv2.imread(image)
    # img_ref = cv2.imread(images[i-1])

    # test_mask = np.ones(mask_template.shape, dtype=np.uint8)
    # EXTRACT FEATURES
    kp_template, desc_template = sift.detectAndCompute(img_template, mask=mask_template)
    # kp_ref, desc_ref = sift.detectAndCompute(img_ref, mask=mask_reference)

    # if len(kp_template) == 0 or len(kp_ref) == 0:
    #     continue
    # sift_kp, sift_desc = sift.compute(img, sift_kp)

    # MATCH FEATURES
    # matches = bf.match(desc_template, desc_ref)
    # if matches is None:
    #     continue
    # matches = sorted(matches, key = lambda x:x.distance)

    # final_img = cv2.drawMatches(img_template, kp_template, img_ref, kp_ref,
    #                             matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # draw the keypoints

    final_img = cv2.drawKeypoints(img_template, kp_template, None,
                                  color=(0,255,0))

    # DRAW LIDAR SMALL OBSTACLE POINTS

    SO_indices = np.where(cluster_preds == -1)[0]
    mask,hull = create_mask(kp_template, img_template.shape,y_features,x_features,SO_indices)
    # polygon = np.zeros((img_template.shape[0],img_template.shape[1]))
    # polygon[hull] = 1
    for i in SO_indices:
        if mask[x_features[i],y_features[i]] == 1:
            cv2.circle(final_img, (y_features[i], x_features[i]), 3, color=(0,0,255))

    cv2.imshow('feed', final_img)
    # if cv2.waitKey(wk) == ord('q'):
    #     break
    break