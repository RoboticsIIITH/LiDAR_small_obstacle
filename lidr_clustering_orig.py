from termcolor import colored
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data
import os
import numpy as np
from PIL import Image
import scipy.cluster as cluster
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor,NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,HuberRegressor,RANSACRegressor,TheilSenRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy import interpolate
import math
import warnings
warnings.filterwarnings("ignore")

def main(draw=True, path=False, paths=None):


    if not path:
        # absolute path
        path = "/media/ash/OS/small_obstacle_bag/synced_data/seq_1/"
        label_files = sorted(os.listdir(os.path.join(path,"labels")))
        velodyne_files = sorted(os.listdir(os.path.join(path,"depth")))

        # path files for various interesting data streams
        index_value = 25
        cloud_files = [os.path.join(path+"depth/",x) for x in label_files]
        image_files = [os.path.join(path+"image/",x) for x in label_files]
        ring_files = [os.path.join(path+'rings', x) for x in label_files]
        label_files = [os.path.join(path+"labels/",x) for x in label_files]

        depth_file = cloud_files[index_value]
        image_file = image_files[index_value]
        label_file = label_files[index_value]
        ring_file = ring_files[index_value]
    else:
        depth_file = paths['depth']
        image_file = paths['image']
        label_file = paths['label']
        ring_file = paths['ring']


    # OPEN ALL RELEVANT IMAGES FOR MANIPULATION
    # open a depth image
    depth = Image.open(depth_file)
    # normalize depth values so that they have range between 1 - 255
    depth = np.array(depth)

    # open rgb image
    image = np.asarray(Image.open(image_file))

    # open ground truth image
    gt = Image.open(label_file)
    gt_unconverted = np.asarray(gt)
    gt = gt.convert('L')
    gt = np.array(gt)

    # RETIRIEVE ROAD POLYGONS FROM THE GROUND TRUTH FILE
    # retrieve max and min of these pixels
    road_x,road_y = np.where(gt == 38)
    max_road_x = np.max(road_x)
    min_road_x = np.min(road_x)
    max_road_y = np.max(road_y)
    min_road_y = np.min(road_y)

    # open a ring image
    ring = np.asarray(Image.open(ring_file))
    # CL
    # CONTROL VARIABLES THAT ARE GOING TO BE USED TO ANALYSE LIDAR DATA
    # data fields to be potted
    # x, y self-explanatory
    # Value is the depth value at x,y
    x=[] # a list of all x positions corresponding to lidar positions within the road polygon
    y=[] # list of y positions corresponding to lidar positions within road polygon
    value = [] # vx, y lidar values
    ring_values = [] # ring value of at the position x, y


    # Loop to extract pixels with valid lidar depth that belong within the road
    # polygon
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i,j] == 0:
                continue
            if i<max_road_x and i>min_road_x and j<max_road_y and j>min_road_y and gt[i,j]!=0:
                x.append(i)
                y.append(j)
                value.append(depth[i, j])
                ring_values.append(ring[i, j])


    x = np.asarray(x)
    y = np.asarray(y)
    value = np.asarray(value)
    ring_values = np.asarray(ring_values)
    unique_road_rings = np.unique(ring_values)

    x_features = []
    y_features = []
    value_features = []
    cluster_preds = []

    # degree of linear regression fit
    for ring in unique_road_rings[:len(unique_road_rings)-1]:
        relevant_indices = np.where(ring_values == ring)[0]
        for i in relevant_indices:
            x_features.append(x[i])
            y_features.append(y[i])
            value_features.append(value[i])
        feature_vector = [[x[i],y[i]] for i in relevant_indices]
        ring_depths = value[relevant_indices]
        ring_depths = np.asarray(ring_depths)
        ring_depths = ring_depths[:, np.newaxis]
        # preds = custom_fit(y[relevant_indices][:,np.newaxis],ring_depths)
        # preds = custom_fit(x[relevant_indices][:,np.newaxis],y[relevant_indices][:,np.newaxis])
        preds = spline_fit(x[relevant_indices],y[relevant_indices],value[relevant_indices])
        # clf = IsolationForest(contamination=0.02).fit(feature_vector)
        # preds = clf.predict(feature_vector)
        for pred in preds:
            cluster_preds.append(pred)
        break

    # TRYING OUT VARIOUS CLUSTERING ALGOS
    # build a feature vector to input to various clustering apis
    # lets cluster ring 1

    # depth_ring_1 = np.asarray(depth[x_1_road, y_1_road])

    # ITERATION 1, FEATURE VECTOR USING X,Y,D
    # feature_vector = [[x_1_road[i],y_1_road[i],depth[x_1_road[i], y_1_road[i]]] for i in range(len(x_1_road))]

    # ITERATION 2, USING ONLY DEPTH
    # feature_vector = [[depth_ring_1[i]] for i in range(len(x_1_road))]
    # feature_vector = np.array(feature_vector)
    # norm_feature_vector = cluster.vq.whiten(feature_vector)
    # depth_ring_1 = depth_ring_1[:, np.newaxis]

    # previous clustering code
    # feature_vector_global = [[x[i], y[i], value[i]] for i in range(len(value))]
    # feature_vector_global = np.array(feature_vector_global)
    # norm_feature_vector_global = cluster.vq.whiten(feature_vector_global)
    # value = np.array(value)
    # value = value[:, np.newaxis]


    """k means clustering"""
    # cluster,distortion = cluster.vq.kmeans(norm_feature_vector.transpose(),k_or_guess=2)

    """SVM clustering"""
    # clf = OneClassSVM(gamma='auto',kernel='poly',degree=3).fit(value)
    # pred = clf.predict(value)
    # clf = OneClassSVM(gamma='auto',kernel='poly',degree=3).fit(depth_ring_1)
    # pred = clf.predict(depth_ring_1)

    "Isolation forest clustering"
   #  clf = IsolationForest(contamination=0.1).fit(depth_ring_1)
   #  pred = clf.predict(depth_ring_1)
    # clf = IsolationForest(n_estimators=500, contamination=0.1).fit(norm_feature_vector)
    # pred = clf.predict(norm_feature_vector)

    # clf = LocalOutlierFactor().fit(feature_vector)
    # pred = clf.fit_predict(feature_vector)

    """Nearest Neighbours"""
    # nbrs = NearestNeighbors(n_neighbors=5,radius=100).fit(value)
    # distances, indices = nbrs.kneighbors(value)
    # print(np.argmin(distances,axis=1),distances)

    colors=[]
    for elem in cluster_preds:
        if elem == -1:
            colors.append('red')
        elif elem == 1:
            colors.append('blue')

    if draw:
        fig=plt.figure(dpi=200)
        # ax_ring = fig.add_subplot(111,projection='3d')
        ax_global = fig.add_subplot(111, projection='3d')
        # ax_global.scatter(xs=norm_feature_vector_global[:,1],ys=norm_feature_vector_global[:,2],zs=norm_feature_vector_global[:,0],s=1,c=ring_values)
        ax_global.scatter(xs=y_features,ys=value_features,zs=x_features,s=1,color=colors)
        # ax.scatter(xs=feature_vector[:,0],ys=feature_vector[:,2],zs=feature_vector[:,1],s=1)
        # ax_ring.scatter(xs=norm_feature_vector[:,1],ys=norm_feature_vector[:,2],zs=norm_feature_vector[:,0],s=1, color=colors)
        # ax.scatter(xs=x_1_road,ys=feature_vector[:,0],zs=y_1_road,s=1)

        # print('gt shape: ', gt_unconverted.shape)
        # print('X: ', gt_unconverted.shape[0])
        # print('Y: ', gt_unconverted.shape[1])
        # gt_unconverted = read_png(label_files[25])
        # stepX, stepY = 1000/image.shape[1], 1000/image.shape[0]
        # X1 = np.arange(0, 1000, stepX)
        # Y1 = np.arange(0, 1000, stepY)
        # stepX, stepY = 1, 1
        # X1 = np.arange(0, 1280, 1)
        # Y1 = np.arange(0, 720, 1)
        # X1, Y1 = np.meshgrid(X1, Y1)
        #
        # ax.plot_surface(X1, np.atleast_2d(255), Y1, rstride=10, cstride=10,
        #                 facecolors=image/256.0)
        # ax_ring.set_xlabel('X')
        # ax_ring.set_ylabel('depth')
        # ax_ring.set_zlabel('Z')
        ax_global.set_xlabel('X')
        ax_global.set_ylabel('depth')
        ax_global.set_zlabel('Z')
        # ax.scatter(xs=cluster[:,1],ys=cluster[:,2],zs=cluster[:,0],s=10,color=colors)
        plt.show()

    return np.asarray(x_features), np.asarray(y_features), np.asarray(cluster_preds), np.asarray(value_features)

def custom_fit(x_feat, y_feat):
    result = []
    degree = 3
    model = make_pipeline(PolynomialFeatures(degree),RANSACRegressor())
    # print(colored("DEBUG/ x shape: {}".format(x_feat.shape), 'cyan'))
    # print(colored("DEBUG/ y shape: {}".format(y_feat.shape), 'cyan'))
    model.fit(x_feat, y_feat)
    y_hat = model.predict(x_feat)

    # plt.scatter(x_feat,y_hat,c='r',s=5)
    # plt.scatter(x_feat,y_feat,c='b',s=10)
    # plt.show()
    # print(colored("DEBUG/ y hat shape: {}".format(y_hat.shape), 'cyan'))
    deviation = np.sqrt(np.mean(abs(y_hat - y_feat)**2))
    for i in range(len(y_feat)):
        if abs(y_feat[i] - y_hat[i]) > 2.5*deviation:
            result.append(-1)
        else:
            result.append(1)
    return result

def spline_fit(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(x,y,z)
    plt.show()

def get_breakpoints(pts):
    diff_log = []
    length = pts.shape[0]
    pred = np.zeros(length)
    new_pred = np.zeros(length)
    # road_height = np.mean(pts[:,2])

    for i in range(2,length):
        d_i_1 = np.linalg.norm(pts[i-1,:3])
        d_i_2 = np.linalg.norm(pts[i-2,:3])
        d_i = np.linalg.norm(pts[i,:3])
        gamma_1 = np.dot(pts[i-1,:3],pts[i-2,:3])/(d_i_1*d_i_2)
        gamma_2 = np.dot(pts[i - 1, :3], pts[i, :3]) / (d_i_1 * d_i)
        gamma = (gamma_1 + gamma_2)/2
        d_p = (d_i_1*d_i_2)/(2*d_i_1*gamma-d_i_2)
        diff = d_i-d_p

        if 0.5 < diff < 1:
            pred[i] = 1
        elif -1 < diff < -0.5:
            pred[i] = -1
        diff_log.append(diff)

    min_segment = 1
    segments = []
    for i in range(length):
        if pred[i] == -1:
            obs_start = i
            obs_end = 0
            end_range = i+11 if i+11 < length else length
            for j in range(i+1,end_range):
                if pred[j] == 1 and j-i > min_segment:
                    obs_end = j
                    break
            if obs_start != 0 and obs_end != 0:
                segments.append((obs_start,obs_end))

    for start,end in segments:
        # if np.mean(pts[start:end,:2]) - road_height < 0.5:
        new_pred[start:end] = -1
    return new_pred

if __name__ == '__main__':
    main()

