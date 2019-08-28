from PIL import Image
import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R

path = "/media/ash/OS/small_obstacle_bag/synced_data/seq_1/"
img_path = os.path.join(path,"image")
velo_path = os.path.join(path,"velodyne")

#TODO: create utility function to read from txt file
transform_matrix = [[0.99961240, 0.00960922,-0.02612872,0.257277],
                    [-0.01086974,0.99876225,-0.04853676,-0.0378583],
                    [0.02562997,0.04880196,0.99847958,-0.0483284],
                    [0, 0, 0,1]]


projection_matrix = [[692.653256 ,0.000000, 629.321381],
                     [0.000,692.653256,330.685425],
                     [0.000000,0.000000, 1.00000]]

distortion_matrix = [-0.163186, 0.026619, 0.000410, 0.000569, 0.000000]

transform_matrix=np.array(transform_matrix)
projection_matrix=np.array(projection_matrix)
distortion_matrix=np.array(distortion_matrix)

"""
Transforming co-ordinate axis to LHS system: the one followed in odometry
The transformation matrix above came from a different choice of axis, ignore this transformation 
if they both have same axis 
"""
hacky_trans_matrix = R.from_euler('xyz',[1.57,-1.57,0]).as_dcm()
hacky_trans_matrix = np.concatenate((hacky_trans_matrix,np.zeros(3)[:,np.newaxis]),axis=1)
hacky_trans_matrix = np.concatenate((hacky_trans_matrix,np.array([[0,0,0,1]])),axis=0)

rot_vec = transform_matrix[:3,:3]
# dump_vec = R.from_dcm(rot_vec)
# dump_vec = dump_vec.as_rotvec()
# print("Rot before",dump_vec)
# dump_vec[0] += 0.01
# print("Rot after",dump_vec)
# new_rot_vec = R.from_rotvec(dump_vec)
# new_rot_vec = new_rot_vec.as_dcm()
trans_vec= transform_matrix[:3,3]
# print("Trans before",trans_vec)
# trans_vec[0] += 0.25
# print("Trans after",trans_vec)

for file in sorted(os.listdir(velo_path)):

    points = np.load(os.path.join(velo_path,file))
    shifted_points = points
    shifted_points = shifted_points.transpose()
    shifted_points[3,:] = 1                         # Convert to homogeneous coordinates
    shifted_points = shifted_points[:4,:]
    shifted_points = np.dot(hacky_trans_matrix,shifted_points)
    shifted_points = shifted_points.transpose()[:,:3]
    img = cv2.imread(os.path.join(img_path,file.split('.')[0] + '.png'))
    rot, _ = cv2.Rodrigues(rot_vec)
    proj_points,_ = cv2.projectPoints(shifted_points,rot,trans_vec,projection_matrix,distortion_matrix)

    for i in range(proj_points.shape[0]):
        x = int(proj_points[i][0][0])
        y = int(proj_points[i][0][1])
        depth = np.sqrt(pow(shifted_points[i][0],2) + pow(shifted_points[i][1],2) + pow(shifted_points[i][2],2))
        depth = np.clip(depth,a_min=0,a_max=100)
        if (y < 720 and x < 1280 and x > 0 and y > 0 and depth<15 and shifted_points[i][2]>=0 and shifted_points[i][1]>=0):
            hsv = np.zeros((1, 1, 3)).astype(np.uint8)
            hsv[:, :, 0] = int((depth) / (15) * 159)
            hsv[0, 0, 1] = 255
            hsv[0, 0, 2] = 255
            hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.circle(img,(x,y),3,color=(int(hsv[0,0,0]),int(hsv[0,0,1]),int(hsv[0,0,2])),thickness=1)
    cv2.imshow("window",img)
    cv2.waitKey(10)
cv2.destroyAllWindows()