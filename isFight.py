import cv2 as cv
import numpy as np
import math
import json
import torch
from torch import nn, optim
import time
import os
import sys


photo_dir = 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\examples\\media\\COCO_val2014_000000000192.jpg'
model_dir = 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\models'
sys_dir = 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\build\\python\\openpose\\Debug'
os_dir = ';' + 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\build\\x64\\Debug;' + 'C:\\Users\\weightwater\\Desktop\\final\\openpose\\openpose\\openpose-1.7.0\\build\\bin'
saveImage = True
filePath = r'C:\Users\weightwater\Desktop\result'


try:
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(sys_dir)
    os.environ['PATH']  = os.environ['PATH'] + os_dir
    import pyopenpose as op
    params = dict()
    params["model_folder"] = model_dir
    params["net_resolution"] = '160x80'
    params['num_gpu'] = 1
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    print(e)
    sys.exit(-1)


opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()

def read_vedio(vedioPath, useKeyFrame=False, useOptical=False, cropScale=0.8):
    cap = cv.VideoCapture(vedioPath)
    num_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    channel = 3

    # 读取视频中的所有帧，用作之后计算
    frameSeq = np.zeros((num_frame, height, width, channel), dtype=np.uint8)
    for i in range(num_frame):
        ret, frame = cap.read()
        frameSeq[i] = frame
        # cv2.imshow('test', frame[top:bottom, left:right, :])
        # cv2.waitKey(30)

    # 获取要选择的帧
    index = list(range(num_frame))
    if useKeyFrame:
        index = getKeyFrame(frameSeq, window=10)

    # 获取要截选的区域
    left, right = 0, width
    top, bottom = 0, height
    if useOptical:
        top, bottom, left, right = opticalCalculate(frameSeq, cropScale=cropScale)

    # 读取图片返回图片序列
    # opencv 图片存储数据为uint8
    newFrameSeq = np.zeros((len(index), bottom-top, right-left, channel), dtype=np.uint8)
    # print(newFrameSeq.shape, frameSeq.shape, top, bottom, left, right)
    num = 0
    for i in range(num_frame):
        if i in index:
            newFrameSeq[num] = frameSeq[i, top:bottom, left:right, :]
            num += 1
            
        # cv.imshow('test', frameSeq[i, top:bottom, left:right, :])
        # cv.waitKey(30)

    return newFrameSeq


def poseFromVedio(frameSeq, datum):
    poseSeq = []
    for i, frame in enumerate(frameSeq):
        
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        data = datum.poseKeypoints

        if data is not None:
            poseSeq.append(data.tolist())
        if saveImage:
            cv.imwrite(filePath + '/pose' + str(i) + '.jpg', datum.cvOutputData)
    return poseSeq


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


def pose2Feature(poseSeq):
    featureSeq = []
    for pose in poseSeq:
        tmp = get_x(pose)
        featureSeq.append(tmp)
    return featureSeq


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
    

def long(vedioPath, datum, useKeyFrame=False, useOptical=False):
    tic = time.time()
    frameSeq = read_vedio(vedioPath, useKeyFrame=useKeyFrame, useOptical=useOptical)
    poseSeq = poseFromVedio(frameSeq, datum)
    featureSeq = pose2Feature(poseSeq)
    res = checkViolent(featureSeq)
    toc = time.time()
    return res, toc-tic


def opticalCalculate(frameSeq, cropScale=0.9, resize=(224, 224)):
    num_frame, hight, width, channel = frameSeq.shape
    opt_set = np.zeros((num_frame-1, resize[0], resize[1], 2))
    frame1 = frameSeq[0]
    frame1 = cv.resize(frame1, resize, interpolation=cv.INTER_AREA)
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    for i, frame2 in enumerate(frameSeq[1:]):
        frame2 = cv.resize(frame2, resize, interpolation=cv.INTER_AREA)
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if saveImage:
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imwrite(filePath + '/flow' + str(i) + '.jpg', bgr)

        prvs = next
        opt_set[i] = flow
    
    crop_size = (int(resize[0]*cropScale), int(resize[1]*cropScale))
    optSum = np.sum(np.sqrt(opt_set[..., 0]**2 + opt_set[..., 1]**2), axis=0)

    if saveImage:
        cv.imwrite(filePath + '/optSum.jpg', optSum)

    # filter slight noise by threshold
    thresh = np.mean(optSum)
    optSum[optSum < thresh] = 0

    if saveImage:
        cv.imwrite(filePath + '/optSum2.jpg', optSum)

    # calculate the center of gravity of magnitude map and adding 0.001 to avoid empty value
    x_sum = np.sum(optSum, axis=0) + 0.001
    y_sum = np.sum(optSum, axis=1) + 0.001

    # calculate prob in every row and column
    x_prob = x_sum / np.sum(x_sum)
    y_prob = y_sum / np.sum(y_sum)

    # print(x_prob.shape)

    x, y = 0, 0
    for index, (i, j) in enumerate(zip(x_prob, y_prob)):
        x += index*i
        y += index*j

    x, y = int(x), int(y)

    # avoid to beyond boundaries
    nug_x = crop_size[0] // 2
    nug_y = crop_size[1] // 2
    x = max(nug_x, min(x, resize[0]-nug_x))
    y = max(nug_y, min(y, resize[1]-nug_y))

    left_scale, right_scale, top_scale, bottom_scale = (y-nug_y) / resize[0], (y+nug_y) / resize[0], (x-nug_x) / resize[1], (x+nug_x) / resize[1]

    left, right = int(width*left_scale), int(width*right_scale)
    top, bottom = int(hight*top_scale), int(hight*bottom_scale)

    # print(left_scale, right_scale, top_scale, bottom_scale, x, y, nug_x, nug_y)

    return top, bottom, left, right

    # optSum = cv.normalize(optSum[x-nug_x: x+nug_x, y-nug_y: y+nug_y], None, 0, 255, cv.NORM_MINMAX).astype(np.int8)
    # cv.imshow('optsum', optSum)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def checkViolent(featureSeq, threshold=0.7):
    num_Violent = 0
    for featrues in featureSeq:
        violent = False
        for f in featrues:
            # print(f)
            output = net(torch.tensor(f).view(1, 30).float())
            # print(output)
            _, predicted = torch.max(output, 1)
            # print(predicted[0])
            if predicted[0] == 1:
                violent = True
                break
        if violent:
            num_Violent += 1
    
    # print(num_Violent)

    return num_Violent > len(featureSeq) * threshold


def precess_image(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_image = cv.GaussianBlur(gray_image, (3, 3), 0)
    
    return gray_image

num = 0

def abs_diff(pre_image, curr_image):
    gray_pre = precess_image(pre_image)
    gray_curr = precess_image(curr_image)
    diff = cv.absdiff(gray_pre, gray_curr)
    res, diff = cv.threshold(diff, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    if saveImage:
        global num
        num += 1
        cv.imwrite(filePath + '/grayImage' + str(num) + '.jpg', gray_curr)
        cv.imwrite(filePath + '/bineray' + str(num) + '.jpg', diff)

    cnt_diff = np.sum(diff)
    return cnt_diff


def exponential_smoothing(alpha, s):
    s_temp = [s[0]]
    # print(s_temp)
    for i in range(1, len(s)):
        s_temp.append(alpha * s[i-1] + (1-alpha) * s_temp[i-1])
    return s_temp


def getKeyFrame(frameSeq, window=25, alpha=0.07, smooth=True):
    num_frame = len(frameSeq)
    index = []
    diff = []
    frm = 0
    pre_image = np.array([])
    curr_image = np.array([])

    pre_image = frameSeq[0]

    for i, frame in enumerate(frameSeq, 1):
        curr_image = frame
        diff.append(abs_diff(pre_image, curr_image))
        pre_image = curr_image
    
    if smooth:
        diff = exponential_smoothing(alpha, diff)
    
    diff = np.array(diff)
    mean = np.mean(diff)
    dev = np.std(diff)
    diff = (diff - mean) / dev

    # print('pick index')
    for i, d in enumerate(diff):
        ub = len(diff) - 1
        lb = 0
        if not i-window // 2 < lb:
            lb = i - window//2
        if not i-window // 2 > ub:
            ub = i + window//2

        comp_window = diff[lb:ub]
        if d >= max(comp_window):
            index.append(i)
            if saveImage:
                cv.imwrite(filePath + '/keyFrame' + str(i) + '.jpg', frameSeq[i])

    tmp = np.array(index)
    tmp = tmp + 1
    index = tmp.tolist()
    # print("Extract the Frame Index:" + str(index))
    return index


if __name__ == '__main__':
    # print(sys.argv)
    net = ViolentClassification(30, 200, 300, 100, 2)
    net.load_state_dict(torch.load('./model/version1.pth'))
    testVedio = sys.argv[1]

    res, t = long(testVedio, datum=datum, useKeyFrame=True, useOptical=True)
    # res, t = long(test_vedio, datum=datum, useKeyFrame=True, useOptical=True)
    print(res)
