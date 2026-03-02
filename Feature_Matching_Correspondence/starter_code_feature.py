import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
            Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib

####### Evaluate #######
gt_dir = os.path.abspath('./training/gt_depth_map')

## Input
left_image_dir = os.path.abspath('./training/left')
right_image_dir = os.path.abspath('./training/right')
calib_dir = os.path.abspath('./training/calib')
sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']

## Output
output_file = open("P3_result_train.txt", "a")
output_file.truncate(0)


## Main
for i,sample_name in enumerate(sample_list):

    left_image_path = left_image_dir +'/' + sample_name + '.png'
    right_image_path = right_image_dir +'/' + sample_name + '.png'

    img_left = cv.imread(left_image_path, 0)
    img_right = cv.imread(right_image_path, 0)

    # TODO: Initialize a feature detector

    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector
    sift = cv.SIFT_create(1000)

    # find the keypoints and descriptors with SIFT
    kp_left, des_left = sift.detectAndCompute(img_left,None)
    kp_right, des_right = sift.detectAndCompute(img_right,None)

    # # Visualize the keypoints
    img=cv.drawKeypoints(img_left,kp_left,img_left,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img=cv.drawKeypoints(img_right,kp_right,img_right,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite(f'sift_keypoints_left_{sample_name}.jpg',img)
    cv.imwrite(f'sift_keypoints_right{sample_name}.jpg',img)


    # TODO: Perform feature matching

    # Define matchers
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left,des_right,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    # turn epipolar constraint off
    img = cv.drawMatches(img_left, kp_left, img_right, kp_right, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(f'matches_{sample_name}.jpg',img)
    
    # Stereo matching using SIFT keypoints, fundamental matrix estimation, and epipolar line visualization.
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp_right[m.trainIdx].pt)
            pts1.append(kp_left[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    def drawlines(img1,img2,lines,pts1,pts2):
        r,c = img1.shape
        img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
        img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img_left,img_right,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img_right,img_left,lines2,pts2,pts1)
    # plt.subplot(121),plt.imshow(img5)
    # plt.subplot(122),plt.imshow(img3)
    # plt.show()
    cv.imwrite(f'epipolar_{sample_name}.jpg',img)

    # TODO: Perform outlier rejection
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_left[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_right[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img_left.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img_right = cv.polylines(img_right,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    matches=good
    # Read calibration
    calib_file_path = calib_dir +'/' + sample_name + '.txt'
    frame_calib = read_frame_calib(calib_file_path)
    stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

    # Find disparity and depth
    pixel_u_list = [] # x pixel on left image
    pixel_v_list = [] # y pixel on left image
    disparity_list = []
    depth_list = []

     #Load ground truth depth map for evauluation and comparison 
    depth_map_path = gt_dir+'/' + sample_name + '.png'
    depth_map = cv.imread(depth_map_path,0)
    depth_values = depth_map.astype(np.float32)
    #print(depth_values.shape)

    #print(depth_values)
    gt_list =[]

    for i, match in enumerate(matches):
      	# Calculate disparity
        u_left = kp_left[match.queryIdx].pt[0]  # x-coordinate in left image
        u_right = kp_right[match.trainIdx].pt[0]  # x-coordinate in right image
        
        disparity = u_left - u_right

        # Calculate depth
        depth = (stereo_calib.baseline * stereo_calib.f) / disparity # Depth (Z) = (Baseline * Focal Length) / Disparity

        # Append data to lists
        pixel_u_list.append(u_left)
        pixel_v_list.append(kp_left[match.queryIdx].pt[1])  # y-coordinate in left image
        disparity_list.append(disparity)
        depth_list.append(depth)
        gt_list.append(depth_values[int(kp_left[match.queryIdx].pt[1])][int(kp_left[match.queryIdx].pt[0]) ])

    ground_truth = np.array(gt_list)
    estimated_depth = np.array(depth_list)
    mse = np.sqrt(((ground_truth - estimated_depth) ** 2).mean())
    # Calculate the Root Mean Square Error (RMSE)
    #rmse = np.sqrt(mse)
    # Print the RMSE
    #print(mse)
    #print("RMSE:", rmse)
       

    # Output
    for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
        line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
        output_file.write(line + '\n')
    
    # Draw matches
    img = cv.drawMatches(img_left, kp_left, img_right, kp_right, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img)
    # plt.show()
    cv.imwrite(f'match_outlier{sample_name}.jpg',img)

output_file.close()


#### Testing Test ####


## Input
left_image_dir = os.path.abspath('./test/left')
right_image_dir = os.path.abspath('./test/right')
calib_dir = os.path.abspath('./test/calib')
sample_list = ['000011', '000012', '000013', '000014','000015']

## Output
output_file = open("P3_result.txt", "a")
output_file.truncate(0)

for sample_name in sample_list:
    left_image_path = left_image_dir +'/' + sample_name + '.png'
    right_image_path = right_image_dir +'/' + sample_name + '.png'
    img_left = cv.imread(left_image_path, 0)
    img_right = cv.imread(right_image_path, 0)

    # Initiate SIFT detector
    sift = cv.SIFT_create(1000)
    # find the keypoints and descriptors with SIFT
    kp_left, des_left = sift.detectAndCompute(img_left,None)
    kp_right, des_right = sift.detectAndCompute(img_right,None)

    img=cv.drawKeypoints(img_left,kp_left,img_left,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(img), plt.show()
    cv.imwrite(f'sift_keypoints{sample_name}.jpg',img)
    #######################
    # Define matchers
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_left,des_right,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    img = cv.drawMatches(img_left, kp_left, img_right, kp_right, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(f'matches_{sample_name}.jpg',img)
    
    # Stereo matching using SIFT keypoints, fundamental matrix estimation, and epipolar line visualization.
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp_right[m.trainIdx].pt)
            pts1.append(kp_left[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    def drawlines(img1,img2,lines,pts1,pts2):
        r,c = img1.shape
        img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
        img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img_left,img_right,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img_right,img_left,lines2,pts2,pts1)
    # plt.subplot(121),plt.imshow(img5)
    # plt.subplot(122),plt.imshow(img3)
    # plt.show()

    # TODO: Perform outlier rejection
    if len(good)>10:
        src_pts = np.float32([ kp_left[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_right[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img_left.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img_right = cv.polylines(img_right,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), 10) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    matches=good
    #######################
    # Read calibration
    calib_file_path = calib_dir +'/' + sample_name + '.txt'
    frame_calib = read_frame_calib(calib_file_path)
    stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

    # Find disparity and depth
    pixel_u_list = [] # x pixel on left image
    pixel_v_list = [] # y pixel on left image
    disparity_list = []
    depth_list = []
    matches_filterd=[]
    for i, match in enumerate(matches):
        # Calculate disparity
        u_left = kp_left[match.queryIdx].pt[0]  # x-coordinate in left image
        u_right = kp_right[match.trainIdx].pt[0]  # x-coordinate in right image
        disparity = u_left - u_right
        #print(stereo_calib.baseline)
        
        # Calculate depth
        depth = (stereo_calib.baseline * stereo_calib.f) / disparity # Depth (Z) = (Baseline * Focal Length) / Disparity
        if depth <= 80:
            # Append data to lists
            pixel_u_list.append(u_left)
            pixel_v_list.append(kp_left[match.queryIdx].pt[1])  # y-coordinate in left image
            disparity_list.append(disparity)
            depth_list.append(depth)
            matches_filterd.append(match)
        
    
    # Output
    for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
        line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
        output_file.write(line + '\n')
    
    # Draw matches
    img = cv.drawMatches(img_left, kp_left, img_right, kp_right, matches_filterd, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img)
    # plt.show()
    cv.imwrite(f'match_outlier_{sample_name}.jpg',img)



