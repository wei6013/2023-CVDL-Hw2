import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchsummary
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mp
import random
from torchvision import datasets, transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QPointF
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import Ui_meow

class MyWindow(QMainWindow):
    
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_meow.Ui_MainWindow()  # 創建UI對象
        self.ui.setupUi(self)
        self.ui.pushButton_1.clicked.connect(self.Load_Image)
        self.ui.pushButton_2.clicked.connect(self.Load_Video)
        self.ui.pushButton_3.clicked.connect(self.Background_Subtraction)
        self.ui.pushButton_4.clicked.connect(self.Preprocessing)
        self.ui.pushButton_5.clicked.connect(self.Video_tracking)
        self.ui.pushButton_6.clicked.connect(self.Dimension_reduction)
        self.ui.pushButton_7.clicked.connect(self.Show_model_structure_vgg19)
        self.ui.pushButton_8.clicked.connect(self.Show_accuracy_loss)
        self.ui.pushButton_9.clicked.connect(self.Predict)
        self.ui.pushButton_10.clicked.connect(self.Reset)
        self.ui.pushButton_11.clicked.connect(self.Load_Image_inference)
        self.ui.pushButton_12.clicked.connect(self.Show_Image)
        self.ui.pushButton_13.clicked.connect(self.Show_model_structure_resnet50)
        self.ui.pushButton_14.clicked.connect(self.Show_comparison)
        self.ui.pushButton_15.clicked.connect(self.Inference)
        # self.ui.graphicsView_1.setBackgroundBrush(Qt.black)
        self.image=""
        self.video=""
        self.image_inference=""
    
    
    def Load_Image(self):
        filename,filetype = QFileDialog.getOpenFileName(self,'選擇圖片')
        self.image = filename
    
    def Load_Video(self):
        filename,filetype = QFileDialog.getOpenFileName(self,'選擇影片',"","Video Files (*.mp4 *.avi *.mkv)")
        self.video = filename
    
    def Background_Subtraction(self):
        # 初始化背景分割器
        history = 500  # 過去500幀的記憶
        dist2Threshold = 400.0  # 距離閾值
        subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2Threshold, detectShadows=True)

        video = cv2.VideoCapture(f'{self.video}')

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            # 高斯模糊後去背
            blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)
            fg_mask = subtractor.apply(blurred_image)

            frame = cv2.resize(frame, (320, 240))
            fg_mask = cv2.resize(fg_mask, (320, 240))
            result = cv2.bitwise_and(frame, frame, mask=fg_mask)
            
            # 顯示結果          
            combined_frame = cv2.hconcat([frame,
                                          cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR),
                                          result])
            cv2.imshow('Combined Video', combined_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        video.release()
        cv2.destroyAllWindows()

    def Preprocessing(self):
        video = cv2.VideoCapture(f'{self.video}')
        first_frame=None
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if first_frame is None:
                first_frame=frame
                
                frame = cv2.resize(frame, (600, 400))
                gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray_frame, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
                
                if corners is not None:
                    #轉成(x,y)整數
                    bottom_nose_point = tuple(corners[0][0])
                    bottom_nose_point = (int(bottom_nose_point[0]), int(bottom_nose_point[1]))

                    # 畫紅線
                    cv2.line(frame, (bottom_nose_point[0] - 10, bottom_nose_point[1]),
                            (bottom_nose_point[0] + 10, bottom_nose_point[1]), (0, 0, 255), 2)
                    cv2.line(frame, (bottom_nose_point[0], bottom_nose_point[1] - 10),
                            (bottom_nose_point[0], bottom_nose_point[1] + 10), (0, 0, 255), 2)

                cv2.imshow('Preprocessing', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            
    def Video_tracking(self):
        video = cv2.VideoCapture(f'{self.video}')
        
        ret, frame = video.read()
        first_frame = cv2.resize(frame, (600, 400))
        pre_gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(pre_gray, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        pre_point = corners.reshape(-1, 1, 2)
        
        trajectory = np.zeros_like(first_frame)
        trajectory_color = (0, 100, 255)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (600, 400))
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            #計算光流
            next_point, status, _ = cv2.calcOpticalFlowPyrLK(pre_gray, gray_frame, pre_point, None)
            good_new = next_point[status == 1]
            good_old = pre_point[status == 1]
            
            #draw
            for i,(new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                trajectory = cv2.line(trajectory, (a, b), (c, d), trajectory_color, 2)
                cv2.line(frame, (a - 10, b),
                            (a + 10, b), (0, 0, 255), 2)
                cv2.line(frame, (a, b - 10),
                            (a, b + 10), (0, 0, 255), 2)
                
            result = cv2.add(frame, trajectory)
            cv2.imshow('Tracking', result)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
            pre_gray = gray_frame.copy()
            pre_point = good_new.reshape(-1, 1, 2)
    
    def Dimension_reduction(self):
        img = cv2.imread(self.image)
        img = cv2.resize(img, (400, 400))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normalizer = MinMaxScaler(feature_range=(0, 255))
        gray_img_norm = normalizer.fit_transform(gray_img)
        
        n_components = 1 # Set initial number of components
        reconstruction_error = float('inf')
        threshold_error = 3.0  # Threshold reconstruction error
        
        #去推算n的最小值
        while reconstruction_error > threshold_error and n_components < min(gray_img.shape) :
            pca = PCA(n_components=n_components)
            redu_img = pca.fit_transform(gray_img_norm)
            reconstructed_image = normalizer.inverse_transform(pca.inverse_transform(redu_img))
            
            reconstruction_error = mean_squared_error(gray_img, reconstructed_image)
            # print(n_components)
            n_components += 1

        #找到n值最小值
        n_components -= 1
        print('Final, n= ', n_components)
        
        # 利用得到的n值重建圖片
        pca = PCA(n_components=n_components)
        redu_img = pca.fit_transform(gray_img_norm)
        reconstructed_image = normalizer.inverse_transform(pca.inverse_transform(redu_img))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))

        # 原始
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 灰度
        axes[1].imshow(gray_img, cmap='gray')
        axes[1].set_title('Gray Image')
        axes[1].axis('off')

        # 重建
        axes[2].imshow(reconstructed_image, cmap='gray')
        axes[2].set_title(f'Reconstructed Image, n={n_components}')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    
    def Show_model_structure_vgg19(self):
        
        vgg19_bn=models.vgg19_bn(num_classes=10,weights=False)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #vgg19_bn.to(device)
        torchsummary.summary(vgg19_bn,(3,32,32))
    
    def Show_accuracy_loss(self):
        img = 'Accuracy_Loss.png'
        scene = QGraphicsScene()
        self.ui.graphicsView_1.setScene(scene)
        
        self.pixmap = QPixmap(self.ui.label_1.size())
        self.pixmap.fill(QColor(0, 0, 0, 0))
        self.ui.label_1.setPixmap(self.pixmap)
        self.ui.label_1.show()
        
        # 計算視窗大小和圖片大小
        window_width = self.ui.graphicsView_1.width()
        window_height = self.ui.graphicsView_1.height()

        # 設定 QGraphicsView 調整圖片大小以符合視圖
        self.ui.graphicsView_1.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
        
        pixmap = QPixmap(img)
        pixmap = pixmap.scaled(window_width-2, window_height-2)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
       
    def Reset(self):
        # self.ui.graphicsView_1.setStyleSheet("background-color: black;")

        scene = QGraphicsScene()
        self.ui.graphicsView_1.setScene(scene)

        self.last_point = None
        self.ui.label_2.setText('Predict:')
        
        self.pixmap = QPixmap(self.ui.label_1.size())
        self.pixmap.fill(QColor('black'))
        self.ui.label_1.setPixmap(self.pixmap)
        self.ui.label_1.show()

        # 將繪圖事件連接到GraphicsView物件的事件
        self.ui.graphicsView_1.mousePressEvent = self.mousePressEvent
        self.ui.graphicsView_1.mouseMoveEvent = self.mouseMoveEvent
        self.ui.graphicsView_1.mouseReleaseEvent = self.mouseReleaseEvent
    
    
    def mousePressEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            pos_in_view = self.ui.graphicsView_1.mapFromGlobal(event.globalPos())
            self.last_point = pos_in_view - self.ui.graphicsView_1.rect().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.last_point is not None:
            painter = QPainter(self.pixmap)
            pen = QPen(Qt.white)  # 白色筆
            pen.setWidth(10)  # 筆寬
            painter.setPen(pen)
            
            pos_in_view = self.ui.graphicsView_1.mapFromGlobal(event.globalPos())
            pos_in_pixmap  = pos_in_view - self.ui.graphicsView_1.rect().topLeft()
            painter.translate(self.ui.graphicsView_1.horizontalScrollBar().value(), self.ui.graphicsView_1.verticalScrollBar().value())
            
            painter.drawLine(self.last_point, pos_in_pixmap) # 畫
            self.last_point = pos_in_pixmap
            self.updateLabel()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = None
            
    def updateLabel(self):
        self.ui.label_1.setPixmap(self.pixmap)
        self.ui.label_1.show()
            
    def Predict(self):
        # label_1的pixmap
        pixmap = self.ui.label_1.pixmap()

        if not pixmap.isNull():
            file_path = "predict.png"  
            pixmap.save(file_path)
            
            model = torch.load('./model/Wei_Vgg19-bn_weights.pth', map_location=torch.device('cpu'))
            model.eval()
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307,], std=[0.3081,])
            ])
            image = Image.open("predict.png").convert('L')
            image = transform(image)
            image = image.unsqueeze(0)
            # print("meow")
            # 用自己的模型預測
            with torch.no_grad():
                outputs = model(image)
                # print(outputs)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs,dim=1)
            # 預測結果
            # print("meow")
            classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            class_dict = {i: classes[i] for i in range(len(classes))}
            predicted_class = classes[predicted.item()]
            # print("meow")
            #print(f'Predicted Class: {predicted_class}')
            self.ui.label_2.setText(f'Predict: {predicted_class}')    
            
            plt.figure()
            plt.bar(range(10),probabilities[0])
            plt.xlabel('Class')
            plt.ylabel('Probability(%)')
            plt.xticks(range(10),[class_dict[i] for i in range(10)], rotation=45)
            plt.title("Probability of each class")
            plt.show()
           
    def Show_Image(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        cat_dir = './inference_dataset/Cat'
        dog_dir = './inference_dataset/Dog'
        random_cat_image = os.path.join(cat_dir, random.choice(os.listdir(cat_dir)))
        random_dog_image = os.path.join(dog_dir, random.choice(os.listdir(dog_dir)))
        # 建立一個新的 figure
        plt.figure(figsize=(10, 6))

        # 顯示貓圖片在左邊
        plt.subplot(1, 2, 1)
        cat_img = Image.open(random_cat_image)
        cat_img = transform(cat_img)
        cat_img = np.transpose(cat_img.numpy(), (1, 2, 0))
        plt.imshow(cat_img)
        plt.title('Cat Image')
        plt.axis('off')  # 不顯示坐標軸

        # 顯示狗圖片在右邊
        plt.subplot(1, 2, 2)
        dog_img = Image.open(random_dog_image)
        dog_img = transform(dog_img)
        dog_img = np.transpose(dog_img.numpy(), (1, 2, 0))
        plt.imshow(dog_img)
        plt.title('Dog Image')
        plt.axis('off')  # 不顯示坐標軸

        plt.tight_layout()  # 調整子圖之間的間距
        plt.show()
        
    def Show_model_structure_resnet50(self):
        resnet = models.resnet50()
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        torchsummary.summary(resnet,(3,224,224))
            
    def Show_comparison(self):
        img = mp.imread('Accuracy_comparison.png')
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        """val_accuracy_erasing = np.loadtxt('./model/val_accuracy.txt')
        val_accuracy_noerasing = np.loadtxt('./model/val_accuracy_noerasing.txt')

        max_erasing = max(val_accuracy_erasing)
        max_noerasing = max(val_accuracy_noerasing)
        
        x = ['With Random Erasing', 'Without Random erasing']
        y = [max_erasing, max_noerasing]
        
        #Loss
        plt.figure(figsize=(10, 8))
        bars = plt.bar(x, y, color=['skyblue', 'orange'])
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy comparison')
        plt.ylim(0, 100)
        plt.tight_layout()
        for bar, value in zip(bars, y):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
        
        plt.show()
        plt.savefig('Accuracy_comparison.png', dpi=300)"""

    def Load_Image_inference(self):
        self.image_inference=""
        filename,filetype = QFileDialog.getOpenFileName(self, '選擇圖片')
        if filename:
            self.image_inference=filename
            scene = QGraphicsScene()

            pixmap = QPixmap(filename)
            pixmap=pixmap.scaled(224, 224)
            pixmap_item = QGraphicsPixmapItem(pixmap)

            scene.addItem(pixmap_item)

            self.ui.graphicsView_2.setScene(scene)
    
    def Inference(self):
        self.ui.label_3.setText('Predict:')
        model = torch.load('./model/Wei_ResNet50-erasing_weights.pth', map_location=torch.device('cpu'))
        model.eval()
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(self.image_inference)
        image = transform(image)
        image = image.unsqueeze(0)
        # 用自己的模型預測
        with torch.no_grad():
            outputs = model(image)
            predicted = (outputs >= 0.5).int()  # outputs >= 0.5 -> Dog, othrewise, cat
            print(predicted)
        # 預測結果
        classes = ['cat', 'dog']
        predicted_class = classes[predicted.item()]

        #print(f'Predicted Class: {predicted_class}')
        self.ui.label_3.setText(f'Predict: {predicted_class}')
    
        
if __name__=='__main__':
    app=QApplication(sys.argv)
    MainWindow=MyWindow()
    #ui=Ui_Hw1.Ui_MainWindow()
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())