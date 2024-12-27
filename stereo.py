import cv2
import numpy as np

left = cv2.VideoCapture(0)
right = cv2.VideoCapture(2)

# left_mtx = np.loadtxt("calibration-data/left-mtx.txt")
# left_dist = np.loadtxt("calibration-data/left-dist.txt")
# left_newcameramtx = np.loadtxt("calibration-data/left-newcameramtx.txt")
# left_roi = np.loadtxt("calibration-data/left-roi.txt")
# left_x, left_y, left_w, left_h = left_roi

# right_mtx = np.loadtxt("calibration-data/right-mtx.txt")
# right_dist = np.loadtxt("calibration-data/right-dist.txt")
# right_newcameramtx = np.loadtxt("calibration-data/right-newcameramtx.txt")
# right_roi = np.loadtxt("calibration-data/right-roi.txt")
# right_x, right_y, right_w, right_h = right_roi

Kl = np.array([[814.3892315984838 , 0.0              , 301.7785475156747], 
            [0.0               , 814.3244564142581, 241.14856122951875],
            [0.0               , 0.0              , 1.0]])

Dl = np.array([-0.013341138172127698, 1.031648627418663, -0.0031902153941301176, -0.0037519439307395898, -3.3532131830283163])

Kr = np.array([[938.8610797386298 , 0.0             , 322.6773549802199 ],
               [0.0               , 939.607634474879, 260.0524675860012 ],
               [0.0               , 0.0             , 1.0 ]])

Dr = np.array([0.06315692122168046, 0.6926962517402164, 0.0009153194113601114, -0.0007337065418816848, -4.491416327279584])

R = np.array([[0.9999324233022411, -0.011554149922569324, 0.0012846978144796344],
              [0.01160660470257119, 0.998478843615624, -0.053900701101974806 ],
              [-0.0006599668067333194, 0.05391196965028237, 0.9985454741634159 ]])

T = np.array([-10.360095738719519 ,
              -0.38124143722726817 ,
              -0.4129998284178894, ])

Translation = np.float32([[1, 0, 0], [0, 1, 0]]) 

stereo = cv2.StereoSGBM.create(numDisparities=128, blockSize=21)

img_size = (640, 480)

while True:

    if not left.grab() or not right.grab():
        print("No More Frames")
        break

    _, leftFrame = left.retrieve()
    _, rightFrame = right.retrieve()

    left_img = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
    
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, img_size, R, T)

    xmap1, ymap1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, img_size, cv2.CV_32FC1)
    xmap2, ymap2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, img_size, cv2.CV_32FC1)

    left_img_rectified = cv2.remap(left_img, xmap2, ymap2, cv2.INTER_LINEAR)
    right_img_rectified = cv2.remap(right_img, xmap1, ymap1, cv2.INTER_LINEAR)

    left_img_translated = cv2.warpAffine(left_img_rectified, Translation, img_size) 
    
    disparity = stereo.compute(left_img_translated, right_img_rectified)
    norm_depth = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imshow('left', left_img_translated)
    cv2.imshow('right', right_img_rectified)

    cv2.imshow('depth', norm_depth)

    keypress = cv2.waitKey(1) & 0xff
    if keypress == ord('q'):
        break

def releaseCams():
    left.release()
    right.release()

releaseCams()
    
cv2.destroyAllWindows()
