import numpy as np
import torch
import time

#Approach 1
def approach1(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)
    return dist

#Numerically unstable due to rounding errors

def approach2(x,y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    # print(x_norm.shape,y_norm.shape)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

start = time.time()

# Pretty slow(but torch approach)
# dist = torch.norm(x[:,None] - y, dim=2, p=2)
x = torch.randn(3, 2)
x.requires_grad = True
y = torch.randn(3, 2)
# y.requires_grad = True

# dist = torch.FloatTensor([0])
# dist.requires_grad=True
for i in range(100):
    # dist = approach1(x, y)
    dist=torch.norm(x[:, None] - y, dim=2, p=2)
    dist, _ = torch.min(dist, dim=1)
    dist = torch.max(dist)
    # dist = approach2(x,y)
    dist.backward()
    x.data -= .1*x.grad.data
    print(x.grad)
    x.grad.data.zero_()


z = x*y + y
z,_ = torch.min(z,dim=1)
z = torch.max(z)
z.backward()
print(x.grad,y)

"""
def spline_fit(data):
    # y = data[:,1]
    # z = data[:,2]
    # data[:,1] = z
    # data[:,2] = y

    # tck = interpolate.bisplrep(data[:,0],data[:,1],data[:,2])
    # pred = interpolate.bisplev(sorted(data[:,0]),sorted(data[:,1]),tck)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    model = make_pipeline(PolynomialFeatures(3),HuberRegressor())
    # print(colored("DEBUG/ x shape: {}".format(x_feat.shape), 'cyan'))
    # print(colored("DEBUG/ y shape: {}".format(y_feat.shape), 'cyan'))
    model.fit(np.c_[data[:,0],data[:,2]],data[:,1])
    y_hat = model.predict(np.c_[data[:,0],data[:,2]])
    # y_hat = interpolate.CloughTocher2DInterpolator(np.c_[data[:,0],data[:,2]],data[:,1])
    mse = mean_squared_error(y_hat,data[:,1])
    error = [(y_hat[i] - data[i,1])**2 for i in range(len(y_hat))]
    # sigma = np.std(error)
    # print(sigma)
    colors = []
    outliers = []
    for term in error:
        if term>4*mse:
            colors.append('red')
            outliers.append(-1)
        else:
            colors.append('green')
            outliers.append(1)
    # ax.scatter(data[:, 0], data[:, 2], data[:, 1], c='b', s=10)
    # ax.scatter(data[:, 0], data[:, 2], y_hat, c=colors, s=10)
    # ax.scatter(data[:,0],data[:,2],y_hat-data[:,1],c='g',s=10)

    # ax.scatter(data[:,0],data[:,1],pred[0,:],c='b',s=10)
    # A = np.c_[np.ones(data.shape[0]), data[:, :1], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX=X.flatten()
    YY=Y.flatten()
    # Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
    Z = C[0] * X + C[1] * Y + C[2]
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

    # plt.xlabel('X')
    # plt.ylabel('Z')
    # ax.set_zlabel('Y')
    # plt.show()
    return outliers


def ring_cluster(pts):
    num_clusters=0
    clusters = {num_clusters:[]}
    outliers = [num_clusters]
    angles = []
    median_y = np.mean(pts[:,1])
    for i in range(1,pts.shape[0]):
        angle = np.dot(pts[i-1,:3]-pts[i,:3], pts[i-1,:3])/(np.linalg.norm(pts[i-1,:3]-pts[i,:3])*np.linalg.norm(pts[i-1,:3]))
        # mid_point = pts[i,:3]
        # mid_point2 = pts[i-1,:3]
        # mid_point[1] = median_y
        # angle = np.dot(pts[i,:3],pts[i-1,:3])/(np.linalg.norm(pts[i,:3])*np.linalg.norm(pts[i-1,:3]))
        # angle = np.clip(angle,a_min=-1,a_max=1)
        angle = np.degrees(np.arccos(angle))
        angles.append(angle)

        if angle < 20:
            num_clusters += 1
            clusters[num_clusters] = []
            clusters[num_clusters].append(pts[i])
        else:
            clusters[num_clusters].append(pts[i])
        outliers.append(num_clusters)

    print(clusters.keys())
    # outliers = [3 if x>=3 else x for x in outliers]
    
    outliers = np.array(outliers)
    # print("before",np.unique(outliers))
    for i in range(vinum_clusters+1):
        for j in range(i+1,num_clusters+1):
            # print(abs(np.mean(pts[outliers == i, 1]) - np.mean(pts[outliers == j, 1])))
            # print(np.mean(pts[outliers==i,1]))
            if abs(np.mean(pts[outliers == i,1])-np.mean(pts[outliers == j,1])) < 0.05:
                outliers[outliers==j] = i
            else:
                continue
    # print("after",np.unique(outliers))
    return angles,outliers

"""