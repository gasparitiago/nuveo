from os import wait
import cv2
import glob
import json
import os
from shutil import copyfile


if not os.path.exists('TrainingSetClean'):
    os.makedirs('TrainingSetClean')

if not os.path.exists('ManualConference'):
    os.makedirs('ManualConference')

train_image_path = 'TrainingSet/'
train_files = glob.glob(train_image_path + '*.jpg')
train_files = sorted(train_files)

train_json = glob.glob(train_image_path + '*.json')
train_json = sorted(train_json)

for i in range(len(train_files)):
    image = cv2.imread(train_files[i])
    json_content = json.load(open(train_json[i]))

    points = json_content['shapes'][0]['points']

    if len(points) < 4:
        # don't copy the file to the clean dataset when there aren't 4 points.
        continue

    # Copy files with 4 points to the Clean set
    copyfile(train_files[i], 'TrainingSetClean/' + os.path.basename(train_files[i]))
    copyfile(train_json[i], 'TrainingSetClean/' + os.path.basename(train_json[i]))

    # Create an image with the keypoints to easily identify if there are incorrect annotations
    cv2.circle(image, (points[0][0], points[0][1]), 15, color=(0, 0, 255))
    cv2.circle(image, (points[1][0], points[1][1]), 15, color=(0, 0, 255))
    cv2.circle(image, (points[2][0], points[2][1]), 15, color=(0, 0, 255))
    cv2.circle(image, (points[3][0], points[3][1]), 15, color=(0, 0, 255))

    cv2.line(image, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), (0,255,0), thickness=5)
    cv2.line(image, (int(points[1][0]), int(points[1][1])), (int(points[2][0]), int(points[2][1])), (0,255,0), thickness=5)
    cv2.line(image, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])), (0,255,0), thickness=5)
    cv2.line(image, (int(points[3][0]), int(points[3][1])), (int(points[0][0]), int(points[0][1])), (0,255,0), thickness=5)

    cv2.imwrite('ManualConference/' + os.path.basename(train_files[i]), image)