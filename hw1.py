from sys import int_info
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from time import sleep

import hw1_ui as ui

import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
import numpy as np
import glob
import imutils

class Stitcher(object):
    def __init__(self):
        self.isv3 = imutils.is_cv3()

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        m = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if not m:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = m
        result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,status)

            # return a tuple of the stitched image and the
            # visualization
            return result, vis

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.xfeatures2d.SIFT_create()
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        ####################
        #####    Q1    #####
        #################### 
        self.grid = (11, 8)
        self.ButtonQ1_1.clicked.connect(self.Q1_1)
        self.ButtonQ1_2.clicked.connect(self.Q1_2)
        self.ButtonQ1_3.clicked.connect(self.Q1_3)
        choices = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        self.comboBox_Q1_3.addItems(choices)
        self.ButtonQ1_4.clicked.connect(self.Q1_4)
        self.ButtonQ1_5.clicked.connect(self.Q1_5)
        self.intrinsic_matrix = 0
        self.distortion = 0
        self.rotation_vectors = 0
        self.translation_vectors = 0
        self.Q1()

        ####################
        #####    Q2    #####
        #################### 
        self.ButtonQ2_1.clicked.connect(self.Q2_1)
        self.ButtonQ2_2.clicked.connect(self.Q2_2)

        ####################
        #####    Q3    #####
        #################### 
        self.ButtonQ3_1.clicked.connect(self.Q3_1)
        self.ButtonQ3_2.clicked.connect(self.Q3_2)

        ####################
        #####    Q4    #####
        #################### 
        self.ButtonQ4_1.clicked.connect(self.Q4_1)
        self.ButtonQ4_2.clicked.connect(self.Q4_2)
        self.ButtonQ4_3.clicked.connect(self.Q4_3)

    def Q1(self):
        # glob.glob可以同時獲得這個路徑的所有的匹配文件
        picture = glob.glob('Dataset_CvDl_Hw1/Q1_Image/*.bmp')
        # 設定終止條件，迭代30次或移動0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 儲存圖像點
        objpoints = [] # 3維空間
        picture_points = [] # 2維空間
        
        # 處理每張照片
        for filename in picture:
            # object point 初始化宣告 3表示RBG三個圖片
            objp = np.zeros((self.grid[0] * self.grid[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.grid[0],0:self.grid[1]].T.reshape(-1,2)
            
            # 讀取這個檔名的圖片
            pic = cv2.imread(filename)
            # 將圖片換成灰階
            gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)

            # 找 棋盤的corners   cv2.findChessboardCorners(圖片, (col, row),None)
            # col跟row 是棋盤的線
            ret, corners = cv2.findChessboardCorners(gray, self.grid,None)

            # 如果有找到corner 就把找到的點畫出來
            if ret == True:
                # cornerSubPix(轉灰階的圖片,找到的corners,(col,row),)
                corners2 = cv2.cornerSubPix(gray,corners, self.grid,(-1,-1),criteria)

                # 將計算與找到的圖像點儲存進陣列
                objpoints.append(objp)
                picture_points.append(corners2)

        ret, intrinsic_matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(objpoints, picture_points, gray.shape[::-1],None,None)

        self.intrinsic_matrix = intrinsic_matrix
        self.distortion = distortion
        self.rotation_vectors = rotation_vectors
        self.translation_vectors = translation_vectors

    def Q1_1(self):
        print("-------------Q1_1-------------")
        # glob.glob可以同時獲得這個路徑的所有的匹配文件
        picture = glob.glob('Dataset_CvDl_Hw1/Q1_Image/*.bmp')
        # 設定終止條件，迭代30次或移動0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 處理每張照片
        for filename in picture:
            # 讀取這個檔名的圖片
            pic = cv2.imread(filename)
            pic = cv2.resize(pic, (480, 480))
            # 將圖片換成灰階
            gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)

            # 找 棋盤的corners   cv2.findChessboardCorners(圖片, (col, row),None)
            # col跟row 是棋盤的線
            ret, corners = cv2.findChessboardCorners(gray, self.grid,None)

            # 如果有找到corner 就把找到的點畫出來
            if ret == True:
                # cornerSubPix(轉灰階的圖片,找到的corners,(col,row),)
                corners2 = cv2.cornerSubPix(gray,corners, self.grid,(-1,-1),criteria)

                # 畫出找到的corners
                pic = cv2.drawChessboardCorners(pic, self.grid, corners2,ret)
                print(filename)
                cv2.imshow('Q1_1',pic)
                # 顯示500ms
                cv2.waitKey(500)
        # 關掉這張照片
        cv2.destroyAllWindows()

        print("-------------Q1_1 Finsh-------------\n")
    
    def Q1_2(self):
        print("-------------Q1_2-------------")
        print("Intrinsic Matrix：")
        print(self.intrinsic_matrix,"\n")

        print("-------------Q1_2 Finsh-------------\n")

    def Q1_3(self):
        print("-------------Q1_3-------------")
        # 將下拉選單選到的數字轉乘int
        picture_number = int(self.comboBox_Q1_3.currentText())
        print("you choice picture",picture_number)

        rotation_matrix,_ = cv2.Rodrigues(self.rotation_vectors[picture_number-1])
        Extrinsic_Matrix = np.append(rotation_matrix, self.translation_vectors[picture_number-1],axis=1)
        
        print("Extrinsic Matrix：")
        print(Extrinsic_Matrix,"\n")

        print("-------------Q1_3 Finsh-------------\n")
        
    def Q1_4(self):
        print("-------------Q1_4-------------")
        
        print("Distortion Matrix：")
        print(self.distortion,"\n")
        print("-------------Q1_4 Finsh-------------\n")
        
    def Q1_5(self):
        print("-------------Q1_5-------------")
        # glob.glob可以同時獲得這個路徑的所有的匹配文件
        picture = glob.glob('Dataset_CvDl_Hw1/Q1_Image/*.bmp')

        # 處理每張照片
        for filename in picture:
            # object point 初始化宣告 3表示RBG三個圖片
            objp = np.zeros((self.grid[0] * self.grid[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.grid[0],0:self.grid[1]].T.reshape(-1,2)
            
            # 讀取這個檔名的圖片
            pic = cv2.imread(filename)
            # 重新設定圖片大小
            pic = cv2.resize(pic, (480, 480))

            dst = cv2.undistort(pic, self.intrinsic_matrix, self.distortion)

            # 把兩張照片弄成一個視窗
            display = np.hstack([pic,dst])
            print(filename)
            cv2.imshow('Q1_5',display)

            # 顯示500ms
            cv2.waitKey(500)
        # 關掉這張照片
        cv2.destroyAllWindows()
        print("-------------Q1_5 Finsh-------------\n")
    
    def Q2(self,picture):
        objp = np.zeros((self.grid[0] * self.grid[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.grid[0], 0:self.grid[1]].T.reshape(-1, 2)

        # 儲存圖像點
        objpoints = [] # 3維空間
        picture_points = [] # 2維空間

        gray = None

        for pic in picture:
            gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.grid, None)
            if ret:
                objpoints.append(objp)
                picture_points.append(corners)

        ret, intrinsic_matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(objpoints, picture_points, gray.shape[::-1], None, None)

        return picture_points, intrinsic_matrix, distortion, rotation_vectors, translation_vectors

    def WorkPlay(self, flag, No_word):
        # 設定字母初始位置
        position = [[8,5],[4,5],[1,5],[8,2],[4,2],[1,2]]
        Dict = {}
        workData = []
        
        #讀檔
        if flag == 1:
            with open('Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/lib_onboard.txt') as file:
                workData = file.readlines()
        elif flag == 2:
            with open('Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/lib_vertical.txt') as file:
                workData = file.readlines()
        
        for data in workData:

            count = 0
            alph = data[0]
            axis = []

            for a in data:
                # 檢測字符串是否只由數字組成 
                if a.isdigit():
                    count+=1
                    if count%3 == 1:
                        tmp=[]
                        tmp.append(int(a)+position[No_word][0])
                    elif count%3 == 2:
                        tmp.append(int(a)+position[No_word][1])
                    elif count%3 == 0:
                        axis.append(tmp)
                        tmp.append(int(a)*-1)

            axis = np.float32(axis)
            Dict[alph] = axis
            
        return Dict

    def Q2_1(self):
        print("-------------Q2_1-------------")
        # 取得字母畫寫資料
        # lib_onboard = cv2.FileStorage('Dataset_CvDl_Hw1/Q2_Image/Q2_Lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)

        # onboard_dict = {}
        # words = list(string.ascii_uppercase)
        # for word in words:
        #     onboard_dict[word] = lib_onboard.getNode(word).mat()
        #     print(onboard_dict[word])

        # 把字轉大寫
        input_word = self.lineEdit_Q2.text().upper()

        picture = []
        filename = glob.glob('Dataset_CvDl_Hw1/Q2_Image/*.bmp')
        for picture_name in filename:
            picture.append(cv2.imread(picture_name))

        picture_points, mtx, dis, rot, tran = self.Q2(picture)

        flierames = []
        
        # enumerate 列舉
        for index, image in enumerate(picture):
            pic = image.copy()

            for idx,char in enumerate(input_word):
                Dict = self.WorkPlay(1,idx)
                axis = Dict[char]
                picpts, jac = cv2.projectPoints(axis, rot[index], tran[index], mtx, dis)

                for picture_points in range(1, len(axis),2):
                    pi = tuple(map(int, picpts[picture_points-1].flatten()))
                    pj = tuple(map(int,picpts[picture_points].flatten()))
                    pic = cv2.line(pic, pi, pj, (0, 0, 255), 10)
            
            pic = cv2.resize(pic, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            flierames.append(pic)

        for index in range(len(picture)):
            cv2.imshow('AR on board', flierames[index])
            cv2.waitKey(800)
            cv2.destroyAllWindows()

        print("-------------Q2_1 Finsh-------------\n")

    def Q2_2(self):
        print("-------------Q2_2-------------")
        input_word = self.lineEdit_Q2.text().upper()

        picture = []
        filename = glob.glob('Dataset_CvDl_Hw1/Q2_Image/*.bmp')
        for picture_name in filename:
            picture.append(cv2.imread(picture_name))

        picture_points, mtx, dis, rot, tran = self.Q2(picture)

        flierames = []
        
        # enumerate 列舉
        for index, image in enumerate(picture):
            pic = image.copy()

            for idx,char in enumerate(input_word):
                Dict = self.WorkPlay(2,idx)
                axis = Dict[char]
                picpts, jac = cv2.projectPoints(axis, rot[index], tran[index], mtx, dis)

                for picture_points in range(1, len(axis),2):
                    pi = tuple(map(int, picpts[picture_points-1].flatten()))
                    pj = tuple(map(int,picpts[picture_points].flatten()))
                    pic = cv2.line(pic, pi, pj, (0, 0, 255), 10)

            pic = cv2.resize(pic, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            flierames.append(pic)

        for index in range(len(picture)):
            cv2.imshow('AR on board', flierames[index])
            cv2.waitKey(800)
            cv2.destroyAllWindows()

        print("-------------Q2_2 Finsh-------------\n")
    
    def Q3_1(self):
        print("-------------Q3_1-------------")

        picture_L = cv2.imread('Dataset_CvDl_Hw1/Q3_Image/imL.png',0)  #1彩色 0灰階
        picture_R = cv2.imread('Dataset_CvDl_Hw1/Q3_Image/imR.png',0)  #1彩色 0灰階

        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(picture_L,picture_R)

        # 重新設定圖片大小
        disparity = cv2.resize(disparity, (800, 600))

        disparity = np.dstack((disparity,disparity,disparity,))
        disparity = (disparity + 16) / 4096.0

        cv2.imshow('Disparity Map',disparity)

        print("-------------Q3_1 Finsh-------------\n")
    
    def Q3_2(self):
        print("-------------Q3_2-------------")

        picture_L = cv2.imread('Dataset_CvDl_Hw1/Q3_Image/imL.png',0)  # 0讀成黑白  1讀成彩色
        picture_R = cv2.imread('Dataset_CvDl_Hw1/Q3_Image/imR.png',0)
        # 立體匹配演算法
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(picture_L, picture_R)

        picture_L = cv2.imread('Dataset_CvDl_Hw1/Q3_Image/imL.png',1)
        # namedWindow讓視窗大小可以改變
        cv2.namedWindow('left', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('left', 800, 600)
        cv2.imshow('left', picture_L)
        
        picture_R = cv2.imread('Dataset_CvDl_Hw1/Q3_Image/imR.png',1)
        cv2.namedWindow('right', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('right', 800, 600)
        cv2.imshow('right', picture_R)

        def mouse_click(event, x, y,flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                img = picture_R.copy()
                
                # print(disparity.shape)
                # print(x)
                # print(y)

                disparity_value = disparity[y][x] / 16.0
                if disparity_value <= 0:
                    return None

                right_x = int((x - disparity_value))
                combine_img = cv2.circle(img, (right_x, y), 10,(0, 255, 0), -1)

                cv2.imshow('right', combine_img)
                
        cv2.setMouseCallback('left', mouse_click)

        print("-------------Q3_2 Finsh-------------\n")
    
    def Q4_1(self):
        print("-------------Q4_1-------------")

        picture_L = cv2.imread('Dataset_CvDl_Hw1/Q4_Image/Shark1.jpg',1)  #1是彩色轉灰階
        gray_L = cv2.cvtColor(picture_L,cv2.COLOR_BGR2GRAY)
        picture_R = cv2.imread('Dataset_CvDl_Hw1/Q4_Image/Shark2.jpg',1)  #1是彩色轉灰階
        gray_R = cv2.cvtColor(picture_R, cv2.COLOR_BGR2GRAY)
        
        #  SIFT 特徵選取
        sift = cv2.xfeatures2d.SIFT_create() 
        kp_L = sift.detect(gray_L, None)
        kp_R = sift.detect(gray_R, None)

        kp_L = sorted(kp_L, key = lambda keypoint:keypoint.size, reverse = True)[:200]
        kp_R = sorted(kp_R, key = lambda keypoint:keypoint.size, reverse = True)[:200]
        
        # 圈圖片的特徵點
        draw_L=cv2.drawKeypoints(gray_L, kp_L, picture_L)
        draw_R=cv2.drawKeypoints(gray_R, kp_R, picture_R)
        
        # 將兩張圖片弄成一個視窗顯示
        display = np.hstack([draw_L,draw_R])
        cv2.imshow('image-keypoints', display)
        print("-------------Q4_1 Finsh-------------\n")
    
    def Q4_2(self):
        print("-------------Q4_2-------------")

        picture_L = cv2.imread('Dataset_CvDl_Hw1/Q4_Image/Shark1.jpg',1)  #1是彩色轉灰階
        gray_L = cv2.cvtColor(picture_L,cv2.COLOR_BGR2GRAY)
        picture_R = cv2.imread('Dataset_CvDl_Hw1/Q4_Image/Shark2.jpg',1)  #1是彩色轉灰階
        gray_R = cv2.cvtColor(picture_R, cv2.COLOR_BGR2GRAY)

        #  SIFT 特徵選取
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(picture_L, None)
        kp2, des2 = sift.detectAndCompute(picture_R, None)

        # BFMatcher圖像特徵匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # 應用比率測試
        match = [[m] for m, n in matches if m.distance < 0.5 * n.distance]

        match_img = cv2.drawMatchesKnn(gray_L, kp1, gray_R, kp2, match, None, flags=2)
        cv2.imshow("Matched Keypoints",match_img)

        print("-------------Q4_2 Finsh-------------\n")

    def Q4_3(self):
        print("-------------Q4_3-------------")

        picture_L = cv2.imread('Dataset_CvDl_Hw1/Q4_Image/Shark1.jpg',1)  #1是彩色轉灰階
        picture_R = cv2.imread('Dataset_CvDl_Hw1/Q4_Image/Shark2.jpg',1)  #1是彩色轉灰階

        ###########################################################################################
        # stitcher = cv2.createStitcher(False)
        # # stitcher = cv2.Stitcher(cv2.Stitcher_PANORAMA)
        # print(stitcher)
        # (status, warp_img) = stitcher.stitch((picture_L, picture_R))
        # print(status)
        # print(warp_img)
        # print(cv2.Stitcher_OK)
        
        # if status != cv2.Stitcher_OK:
        #     print("不能拼接圖片, error code = %d" % status)
        #     sys.exit(-1)
        # print("拼接成功.")
        # cv2.imshow('Warp Images', warp_img)
        ###########################################################################################

        # 參考 https://yungyung7654321.medium.com/python%E5%AF%A6%E4%BD%9C%E8%87%AA%E5%8B%95%E5%85%A8%E6%99%AF%E5%9C%96%E6%8B%BC%E6%8E%A5-automatic-panoramic-image-stitching-28629c912b5a
        stitcher = Stitcher()
        (result, vis) = stitcher.stitch([picture_L, picture_R], showMatches=True)
        cv2.imshow("result",result)


        print("-------------Q4_3 Finsh-------------\n")

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())