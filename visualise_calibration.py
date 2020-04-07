from PIL import Image
import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R

path = '/media/ash/OS/IIIT_Labels/test/seq_3/'
img_path = os.path.join(path,"image")
velo_path = os.path.join(path,"velodyne")
depth_path = os.path.join(path, "depth")
label_path = os.path.join(path,"labels")
if not os.path.exists(depth_path):
    os.makedirs(depth_path)


def read_txt(path):
    with open(path,'r') as f:
        rows = f.read().split('\n')[:-1]
        values = [row.split(' ')[:-1] for row in rows]
        transform_matrix = np.array(values,dtype=np.float)
        return transform_matrix


transform_matrix = read_txt('best_transf_mat_2.txt')

# projection_matrix = read_txt('projection_mat.txt')
projection_matrix = [[692.653256 ,0.000000, 629.321381],
                     [0.000,692.653256,330.685425],
                     [0.000000,0.000000, 1.00000]]

camera_matrix = [[709.103066,0.000000,621.543559],
                [0.000000,709.978057,333.677376],
                [0.000000,0.000000,1.000000]]
distortion_matrix = [-0.163186,0.026619,0.000410,0.000569,0.000000]

zed_camera_matrix = [[699.8670043945312, 0.0, 603.5809936523438],
                     [0.0, 699.8670043945312, 332.77801513671875],
                     [0.0, 0.0, 1.0]]
zed_dist = [-0.171875, 0.02449920028448105, 0.0, 0.0, 0.0]
# distortion_matrix = np.zeros(5)


transform_matrix=np.array(transform_matrix)
projection_matrix=np.array(projection_matrix)
# projection_matrix = projection_matrix[:3,:3]
distortion_matrix=np.array(distortion_matrix)
camera_matrix = np.array(camera_matrix)
zed_camera_matrix = np.array(zed_camera_matrix)
zed_dist = np.array(zed_dist)

"""
Transforming co-ordinate axis to LHS system: the one followed in odometry
The transformation matrix above came from a different choice of axis, ignore this transformation
if they both have same axis
"""
hacky_trans_matrix = R.from_euler('xyz',[1.57,-1.57,0]).as_dcm()
hacky_trans_matrix = np.concatenate((hacky_trans_matrix,np.zeros(3)[:,np.newaxis]),axis=1)
hacky_trans_matrix = np.concatenate((hacky_trans_matrix,np.array([[0,0,0,1]])),axis=0)

rot_vec = transform_matrix[:3,:3]
trans_vec= transform_matrix[:3,3]


def project_lid_on_img(lid_pt,T,p):
    tran_pt =  np.dot(T,lid_pt)
    proj_lid_pt = np.dot(p,tran_pt).reshape(3,-1)
    pix = np.array([proj_lid_pt[0]/proj_lid_pt[2],proj_lid_pt[1]/proj_lid_pt[2]]).reshape(2,-1)
    return pix


for file in sorted(os.listdir(label_path)):
    if file == ".DS_Store":
        continue
    points = np.load(os.path.join(velo_path,file.split('.')[0] + '.npy'))
    homo_points = points
    homo_points = homo_points.transpose()
    homo_points[3,:] = 1                         # Convert to homogeneous coordinates
    shifted_points = homo_points[:4,:]
    shifted_points = np.dot(hacky_trans_matrix,shifted_points)
    # shifted_points = shifted_points[:,:3]
    # proj_points = project_lid_on_img(shifted_points,transform_matrix,projection_matrix)
    # proj_points = proj_points.transpose()
    shifted_points = shifted_points.transpose()[:,:3]

    img = cv2.imread(os.path.join(img_path,file))
    # img = cv2.undistort(img,cameraMatrix=zed_camera_matrix,distCoeffs=zed_dist)
    # depth_read = Image.open(os.path.join(depth_path,file.split('.')[0] + '.png'))
    # label_read = Image.open(os.pInput camera matrix ￼ath.join(label_path,file.split('.')[0] + '.png'))
    # label_read = np.array(label_read)
    # depth_read = np.array(depth_read)
    depth_img = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint16)
    # if label_read is None:
    #     continue
    rot, _ = cv2.Rodrigues(rot_vec)
    proj_points,_ = cv2.projectPoints(shifted_points,rot,trans_vec,projection_matrix,distCoeffs=np.zeros(5))
    proj_points = proj_points.squeeze()
    # indexes = np.nonzero(depth_read)
    for i in range(proj_points.shape[0]):
        x = int(proj_points[i,0])
        y = int(proj_points[i,1])
        depth = np.sqrt(pow(shifted_points[i][0],2) + pow(shifted_points[i][1],2) + pow(shifted_points[i][2],2))
        depth = np.clip(depth,a_min=0,a_max=100)
        depth = (depth/100)*(65535-10) + 10
        if (0 < y < 720 and 0 < x < 1280 and shifted_points[i][2] >= 0):
            # depth_img[y, x] = depth
            # hsv = np.zeros((1, 1, 3)).astype(np.uint8)
            # hsv[:, :, 0] = int((depth) / (15) * 159)
            # hsv[0, 0, 1] = 255
            # hsv[0, 0, 2] = 200
            # hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # cv2.circle(img,(x,y),1,color=(int(hsv[0,0,0]),int(hsv[0,0,1]),int(hsv[0,0,2])),thickness=1)
            cv2.circle(img,(x,y),2,color=(0,255,0),thickness=1)
    # x,y = indexes
    # for i in range(len(x)):
    #     if label_read[x[i],y[i]] == 2:
    #         cv2.circle(img,(y[i],x[i]),3,color=(0,255,0))

    if img is not None:
        cv2.imshow("window",img)
    # else:
    #     continue
    # cv2.imwrite(os.path.join(depth_path,file),depth_img)
    cv2.waitKey(0)
    # if cv2.waitKey(10) == ord('q'):
    #     print('Quitting....')
    #     break
cv2.destroyAllWindows()