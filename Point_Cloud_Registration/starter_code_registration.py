import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math

def load_point_cloud(path):
    # Load the point cloud data (do NOT change this function!)
    data = pd.read_csv(path, header=None)
    point_cloud = data.to_numpy()
    return point_cloud


def nearest_search(pcd_source, pcd_target):
    # TODO: Implement the nearest neighbour search
    # TODO: Compute the mean nearest euclidean distance between the source and target point cloud
    corr_target = []
    corr_source = []
    ec_dist_mean = 0
    ec_dist_sum = 0

    for p in pcd_source:
        min_d = float('inf')
        closest_point = None

        for q in pcd_target:
            distance = np.linalg.norm(p - q)
            #distance=math.dist(p,q)
            if distance < min_d:
                min_d = distance
                closest_point = q

        corr_source.append(p)
        corr_target.append(closest_point)
        ec_dist_sum= ec_dist_sum + min_d

    ec_dist_mean = ec_dist_sum / len(pcd_source)
    return corr_source, corr_target, ec_dist_mean


def estimate_pose(corr_source, corr_target):
    # TODO: Compute the 6D pose (4x4 transform matrix)
    # TODO: Get the 3D translation (3x1 vector)

    # compute centroid
    centroid_source = np.mean(corr_source, axis=0)
    centroid_target = np.mean(corr_target, axis=0)

    # Subtract centroids to center the point clouds
    centered_source = corr_source - centroid_source
    centered_target = corr_target - centroid_target

    # Compute outer product matrix 
    H = np.dot(centered_source.T, centered_target)

    # Perform SVD on H to find the rotation matrix R
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Handle reflections (ensures proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector t
    t = centroid_target - np.dot(R, centroid_source)

    pose = np.identity(4)
    pose[:3, :3] = R
    pose[:3, 3] = t

    translation_x = t[0]
    translation_y = t[1]
    translation_z = t[2]

    #translation_x, translation_y, translation_z = t
    
    return pose, translation_x, translation_y, translation_z


def icp(pcd_source, pcd_target):
    # TODO: Put all together, implement the ICP algorithm
    # TODO: Use your implemented functions "nearest_search" and "estimate_pose"
    # TODO: Run 30 iterations
    # TODO: Show the plot of mean euclidean distance (from function "nearest_search") for each iteration
    # TODO: Show the plot of pose translation (from function "estimate_pose") for each iteration

    # Initialize the pose as an identity matrix
    #pose = np.identity(4)
    # Initialize variables to store results
    poses = []  # To store the estimated poses
    mean_distances = []  # To store the mean Euclidean distances
    t_x=[]
    t_y=[]
    t_z=[]
    # Number of iterations for ICP
    num_iterations = 30
    for iteration in range(num_iterations):
        # Find the closest point correspondences using nearest_search
        corr_source, corr_target, ec_dist_mean = nearest_search(pcd_source, pcd_target)
        mean_distances.append(ec_dist_mean)

        # Estimate the pose using estimate_pose
        pose, translation_x, translation_y, translation_z = estimate_pose(corr_source, corr_target)
        # Store the results
        poses.append(pose)
        t_x.append(translation_x)
        t_y.append(translation_y)
        t_z.append(translation_z)
        #Apply Tranform matric to the source data points
        pts = np.vstack([np.transpose(corr_source), np.ones(len(corr_source))])
        corr_source = np.matmul(pose, pts)
        corr_source= np.transpose(corr_source[0:3, :])
        pcd_source=corr_source
    
    # Visualize the mean Euclidean distances over iterations
    plt.figure()
    plt.title('Mean Euclidean Distance')
    plt.plot(mean_distances)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Distance')
    plt.legend(loc='upper right')
    #plt.savefig('ICP Loss Plot.jpg')  
    plt.show()

    plt.figure()
    plt.title('Pose Translation')
    plt.plot(t_x, label='X')
    plt.plot(t_y, label='Y')
    plt.plot(t_z, label='Z')
    plt.xlabel('Iteration')
    plt.ylabel('Translation')
    plt.legend(loc='upper right')  
    #plt.savefig('Plot of the estimated 3D translation.jpg')  
    plt.show()

    pose=np.eye(4)
    for t in poses:
        pose = np.matmul(t, pose)

    return pose




def main():
    # Dataset and ground truth poses
    #########################################################################################
    # Training and test data (3 pairs in total)
    train_file = ['bunny', 'dragon']
    test_file = ['armadillo']

    # Ground truth pose (from training data only, used for validating your implementation)
    GT_poses = []
    gt_pose = [0.8738,-0.1128,-0.4731,24.7571,
            0.1099,0.9934,-0.0339,4.5644,
            0.4738,-0.0224,0.8804,10.8654,
            0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    gt_pose = [0.7095,-0.3180,0.6289,46.3636,
               0.3194,0.9406,0.1153,3.3165,
               -0.6282,0.1191,0.7689,-6.4642,
               0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    #########################################################################################



    # Training (validate your algorithm)
    ##########################################################################################################
    for i in range(2):
        # Load data
        path_source = './training/' + train_file[i] + '_source.csv'
        path_target = './training/' + train_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)
        gt_pose_i = GT_poses[i]

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.show()



        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target)

        # Transform the point cloud
        # TODO: Replace the ground truth pose with your computed pose and transform the source point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        # TODO: Evaluate the rotation and translation error of your estimated 6D pose with the ground truth pose
        gtp=GT_poses[i]
        error=gtp-pose
        print(error)

        # Visualize the point clouds after the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration')
        plt.savefig(f'Point Clouds After Registration_{i}.jpg') 
        plt.show()
    ##########################################################################################################



    # Test
    ####################################################################################
    for i in range(1):
        # Load data
        path_source = './test/' + test_file[i] + '_source.csv'
        path_target = './test/' + test_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.show()

        # TODO: Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target)

        # TODO: Show your outputs in the report
        # TODO: 1. Show your estimated 6D pose (4x4 transformation matrix)
        # TODO: 2. Visualize the registered point cloud and the target point cloud
        print(pose)

        # Transform the point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        # Visualize the point clouds after the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], cmap='Greens')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], cmap='Reds')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        plt.savefig('Point Clouds After Registration_test.jpg') 
        ax.set_title('Point Clouds After Registration')
        plt.show()

        

if __name__ == '__main__':
    main()