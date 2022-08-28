import cv2 as cv
import numpy as np
import math
import json
import torch
from torch import nn, optim
import sys
import os

photo_dir = 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\examples\\media\\COCO_val2014_000000000192.jpg'
model_dir = 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\models'
sys_dir = 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\build\\python\\openpose\\Debug'
os_dir = ';' + 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\build\\x64\\Debug;' + 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\build\\bin'

try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(sys_dir)
    os.environ['PATH']  = os.environ['PATH'] + os_dir
    import pyopenpose as op
    params = dict()
    params["model_folder"] = model_dir
    params["net_resolution"] = '160x80'
    params['num_gpu'] = 1

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    print(e)
    sys.exit(-1)



def read_json(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    data = json.loads(data)
    res = []
    for item in data['people']:
        tmp = [item['pose_keypoints_2d'][i*3: i*3+3] for i in range(len(item['pose_keypoints_2d']) // 3)]
        
        counts0 = sum(1 if p == 0 else 0 for p in [item['pose_keypoints_2d']][0])
        if counts0 > 10:
            continue
        res.append(tmp)
        
    return res


def pointDistance(keyPoint):
    """
    :param keyPoint:
    :return:list
    :distance:
    """
    distance0 = (keyPoint[4][0] - keyPoint[9][0]) ** 2 + (keyPoint[4][1] - keyPoint[9][1]) ** 2
    distance1 = (keyPoint[7][0] - keyPoint[12][0]) ** 2 + (keyPoint[7][1] - keyPoint[12][1]) ** 2
    distance2 = (keyPoint[2][0] - keyPoint[4][0]) ** 2 + (keyPoint[2][1] - keyPoint[4][1]) ** 2
    distance3 = (keyPoint[5][0] - keyPoint[7][0]) ** 2 + (keyPoint[5][1] - keyPoint[7][1]) ** 2
    distance4 = (keyPoint[0][0] - keyPoint[4][0]) ** 2 + (keyPoint[0][1] - keyPoint[4][1]) ** 2
    distance5 = (keyPoint[0][0] - keyPoint[7][0]) ** 2 + (keyPoint[0][1] - keyPoint[7][1]) ** 2
    distance6 = (keyPoint[4][0] - keyPoint[10][0]) ** 2 + (keyPoint[4][1] - keyPoint[10][1]) ** 2
    distance7 = (keyPoint[7][0] - keyPoint[13][0]) ** 2 + (keyPoint[7][1] - keyPoint[13][1]) ** 2
    distance8 = (keyPoint[4][0] - keyPoint[7][0]) ** 2 + (keyPoint[4][1] - keyPoint[7][1]) ** 2
    distance9 = (keyPoint[11][0] - keyPoint[14][0]) ** 2 + (keyPoint[11][1] - keyPoint[14][1]) ** 2
    distance10 = (keyPoint[10][0] - keyPoint[13][0]) ** 2 + (keyPoint[10][1] - keyPoint[13][1]) ** 2
    distance11 = (keyPoint[6][0] - keyPoint[10][0]) ** 2 + (keyPoint[6][1] - keyPoint[10][1]) ** 2
    distance12 = (keyPoint[3][0] - keyPoint[13][0]) ** 2 + (keyPoint[3][1] - keyPoint[13][1]) ** 2
    distance13 = (keyPoint[4][0] - keyPoint[23][0]) ** 2 + (keyPoint[4][1] - keyPoint[23][1]) ** 2
    distance14 = (keyPoint[7][0] - keyPoint[20][0]) ** 2 + (keyPoint[7][1] - keyPoint[20][1]) ** 2

    return [distance0, distance1, distance2, distance3, distance4, distance5, distance6, distance7,
            distance8, distance9, distance10, distance11, distance12, distance13, distance14]


def pointAngle(keyPoint):
    angle0 = __myAngle(keyPoint[2], keyPoint[3], keyPoint[4])
    angle1 = __myAngle(keyPoint[5], keyPoint[6], keyPoint[7])
    angle2 = __myAngle(keyPoint[9], keyPoint[10], keyPoint[11])
    angle3 = __myAngle(keyPoint[12], keyPoint[13], keyPoint[14])
    angle4 = __myAngle(keyPoint[3], keyPoint[2], keyPoint[1])
    angle5 = __myAngle(keyPoint[6], keyPoint[5], keyPoint[1])
    angle6 = __myAngle(keyPoint[10], keyPoint[8], keyPoint[13])
    angle7 = __myAngle(keyPoint[7], keyPoint[12], keyPoint[13])
    angle8 = __myAngle(keyPoint[4], keyPoint[9], keyPoint[10])
    angle9 = __myAngle(keyPoint[4], keyPoint[0], keyPoint[7])
    angle10 = __myAngle(keyPoint[4], keyPoint[8], keyPoint[7])
    angle11 = __myAngle(keyPoint[1], keyPoint[8], keyPoint[13])
    angle12 = __myAngle(keyPoint[1], keyPoint[8], keyPoint[10])
    angle13 = __myAngle(keyPoint[4], keyPoint[1], keyPoint[8])
    angle14 = __myAngle(keyPoint[7], keyPoint[1], keyPoint[8])

    return [angle0, angle1, angle2, angle3, angle4, angle5, angle6, angle7,
            angle8, angle9, angle10, angle11, angle12, angle13, angle14]


def __myAngle(A, B, C):
    c = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    a = math.sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2)
    b = math.sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2)
    if 2 * a * c != 0:
        return (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    return 0


def get_dis_ang(data):
    # data = read_json(json_path)
    res = []
    for people in data:
        distance = pointDistance(people)
        angle = pointAngle(people)
        res.append((distance, angle))

    return res


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# input need pose list contained pose
def get_x(data):
    # dis_ang = get_dis_ang(json_path)
    dis_ang = get_dis_ang(data)
    t = []
    for dis, ang in dis_ang:
        st_dis = standardization(np.array(dis))
        
        t_dis = torch.from_numpy(st_dis)
        t_ang = torch.tensor(st_dis)
        tmp = torch.cat((t_dis, t_ang))
        tmp = tmp.view(len(tmp), 1)
        t.append(tmp)
        
        
    return t


def read_vedio(vedioPath, useKeyFrame=False, useOptical=False):
    cap = cv.VideoCapture(vedioPath)
    num_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # 获取要选择的帧
    index = list(range(num_frame))
    if useKeyFrame:
        pass

    # 获取要截选的区域
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    left, right = 0, width
    top, bottom = 0, height
    if useOptical:
        pass

    # 读取图片返回图片序列
    channel = 3
    # opencv 图片存储数据为uint8
    frameSeq = np.zeros((len(index), bottom-top, right-left, channel), dtype=np.uint8)
    num = 0
    for i in range(num_frame):
        if i in index:
            ret, frame = cap.read()
            frameSeq[num] = frame[top:bottom, left:right, :]
            num += 1
        # cv2.imshow('test', frame[top:bottom, left:right, :])
        # cv2.waitKey(30)

    return frameSeq


def poseFromVedio(frameSeq, datum):
    poseSeq = []
    for frame in frameSeq:
        
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        data = datum.poseKeypoints

        if data is not None:
            poseSeq.append(data.tolist())
    return poseSeq


def pose2Feature(poseSeq):
    featureSeq = []
    for pose in poseSeq:
        tmp = get_x(pose)
        featureSeq.append(tmp)
    return featureSeq


def checkViolent(featureSeq, net, threshold=0.7):
    num_Violent = 0
    for featrues in featureSeq:
        violent = 0
        for f in featrues:
            # print(f)
            output = net(torch.tensor(f).view(1, 30).float())
            _, predicted = torch.max(output, 1)
            # print(predicted[0])
            if predicted[0] == 1:
                violent = True
                break
        if violent:
            num_Violent += 1
    
    return num_Violent > len(featureSeq) * threshold


class ViolentClassification(nn.Module):
    # 30 200 300 100 2
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, out_dim):
        super(ViolentClassification, self).__init__()
        self.fc1 = nn.Linear(in_dim, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, out_dim)

        self.relu = nn.ReLU()

    def forward(self, data):
        data = self.relu(self.fc1(data))
        data = self.relu(self.fc2(data))
        data = self.relu(self.fc3(data))
        output = self.fc4(data)

        return output


def long(vedioPath, datum, net, useKeyFrame=False, useOptical=False):
    frameSeq = read_vedio(vedioPath, useKeyFrame=useKeyFrame, useOptical=useOptical)
    poseSeq = poseFromVedio(frameSeq, datum)
    featureSeq = pose2Feature(poseSeq)
    res = checkViolent(featureSeq, net)
    return res


def main():
    test_vedio = r'C:\Users\weightwater\Desktop\final\dataset\data\Videos\Normal_00004.mp4'

    datum = op.Datum()

    net = ViolentClassification(30, 200, 300, 100, 2)
    net.load_state_dict(torch.load('./model/version1.pth'))
    
    res = long(test_vedio, datum, net)
    print(res)


if __name__ == '__main__':
    main()
