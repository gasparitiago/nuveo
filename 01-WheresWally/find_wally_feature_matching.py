import glob
import numpy as np
import cv2 as cv
import os


orb = cv.ORB_create()
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

reference_image = cv.imread('ReferenceData/wally.jpg')
reference_image = cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)

kp1, des1 = orb.detectAndCompute(reference_image, None)

test_image_path = 'TestSet/'
test_files = glob.glob(test_image_path + '*.jpg')
test_files = sorted(test_files)

result_file_csv = 'results_feature_matching.csv'
csv_file = open(result_file_csv, "w")

for test_file in test_files:
    test_image = cv.imread(test_file)
    test_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
    
    kp2, des2 = orb.detectAndCompute(test_image, None)

    knn_matches = matcher.match(des1, des2)
    good_matches = knn_matches
    
    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = kp1[good_matches[i].queryIdx].pt[0]
        obj[i,1] = kp1[good_matches[i].queryIdx].pt[1]
        scene[i,0] = kp2[good_matches[i].trainIdx].pt[0]
        scene[i,1] = kp2[good_matches[i].trainIdx].pt[1]
    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)

    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = reference_image.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = reference_image.shape[1]
    obj_corners[2,0,1] = reference_image.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = reference_image.shape[0]
    scene_corners = cv.perspectiveTransform(obj_corners, H)

    tl = [scene_corners[0, 0, 0], scene_corners[0, 0, 1]]
    tr = [scene_corners[1, 0, 0], scene_corners[1, 0, 1]]
    bl = [scene_corners[2, 0, 0], scene_corners[2, 0, 1]]
    br = [scene_corners[3, 0, 0], scene_corners[3, 0, 1]]

    center = [(tl[0] + tr[0] + bl[0] + br[0]) / 4,
              (tl[1] + tr[1] + bl[1] + br[1]) / 4]

    print(os.path.basename(test_file) + ',' + str(center[0]) + ',' + str(center[1]))
    csv_file.write(os.path.basename(test_file) + ',' + str(center[0]) + ',' + str(center[1]) + '\n')

print('Created file: ' + result_file_csv)
csv_file.close()