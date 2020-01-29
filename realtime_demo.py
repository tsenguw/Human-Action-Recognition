# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:07:39 2020

@author: mb207
"""

from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
from YOLO_API import YOLO

yolo = YOLO()
model = load_model('model/LSTM_1105.h5')
modelb = load_model('model/Behavior_0924.h5')

# %%
from openni import openni2
openni2.initialize('openNI/GIGABYTE DigitalSignage_OpenNI2_driver/Windows/OpenNI2/x64')

# can also accept the path of the OpenNI redistribution
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
mode = openni2.VideoMode(pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM, 
						 resolutionX = 80, resolutionY = 60, fps = 30)
depth_stream.set_video_mode(mode)
depth_stream.start()

#%%
import math
from pykalman import KalmanFilter
import time
from PIL import Image, ImageDraw, ImageFont        

def toWorld(frame,p):
    X,Y,Z = openni2.convert_depth_to_world(depth_stream,p[1],p[0],frame[p[0],p[1]])
    
    x = np.radians(40)
    y = np.radians(0)
    z = np.radians(0)
    
    r11 = math.cos(y)*math.cos(z)-math.sin(x)*math.sin(y)*math.sin(z)
    r13 = math.sin(y)*math.cos(z)+math.sin(x)*math.cos(y)*math.sin(z)
    r21 = math.cos(y)*math.sin(z)+math.sin(x)*math.sin(y)*math.cos(z)
    r23 = math.sin(y)*math.sin(z)-math.sin(x)*math.cos(y)*math.cos(z)    
    R = [[r11, -math.cos(x)*math.sin(z), r13],
         [r21, math.cos(x)*math.cos(z), r23],
         [-math.cos(x)*math.sin(y), math.sin(x), math.cos(x)*math.cos(y)]]
    R = np.array(R)    
    
    return np.matmul([X, Y, Z], R)

def Kalman(data):
    kf = KalmanFilter(n_dim_obs = 1,
                      n_dim_state = 1,
                      initial_state_mean = data[0],
                      initial_state_covariance = 20,
                      transition_matrices = [1],
                      transition_covariance = np.eye(1),
                      transition_offsets = None,
                      observation_matrices = [1],
                      observation_covariance = 10
                     )
    mean,cov = kf.filter(data)
    return mean
        
def heatmap(img):
    im_color = cv2.applyColorMap(
        255-(img/np.max(img)*255).astype(np.uint8), cv2.COLORMAP_OCEAN)
    return im_color

def IOU(one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h):
    lu_x_inter = max((one_x - (one_w / 2)), (two_x - (two_w / 2)))
    lu_y_inter = min((one_y + (one_h / 2)), (two_y + (two_h / 2)))
     
    rd_x_inter = min((one_x + (one_w / 2)), (two_x + (two_w / 2)))
    rd_y_inter = max((one_y - (one_h / 2)), (two_y - (two_h / 2)))
     
    inter_w = abs(rd_x_inter - lu_x_inter)
    inter_h = abs(lu_y_inter - rd_y_inter)
     
    inter_square = inter_w * inter_h
    union_square = (one_w * one_h) + (two_w * two_h) - inter_square 
    if union_square != 0:
        iou = inter_square / union_square 
    else:
        iou = 0
    return iou

def get_mode(arr):
    arr_appear = dict((a, arr.count(a)) for a in arr)
    for k, v in arr_appear.items():
        if v == max(arr_appear.values()):
            mode = k
#    mode = " ".join(mode)
    return mode

def Top(arraylist,k):                 
    maxlist=[]         
    for i in range(0,k):
        maxlist.append(arraylist[i])   
    maxlist.sort(reverse=True)     
    for i in range(k,len(arraylist),1):
        if arraylist[i]>maxlist[k-1]:
            maxlist.pop()
            maxlist.append(arraylist[i])
            maxlist.sort(reverse=True)
    return maxlist

def cv2ImgAddText(img, text, left, top, textColor, textSize):
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def UI(P,A,B,Bp,C,Cp):
    import tkinter as tk  
    window = tk.Tk()
    window.geometry('250x250')  
    
    var = tk.StringVar()  
    l = tk.Label(window, width=20, text= P +' → ', font=('微軟正黑體', 16))
    l.pack()
    
    def print_selection():
        l.config(text= P + ' → ' + var.get())
    def close():
        window.destroy()
    
    r1 = tk.Radiobutton(window, text=A, font=('微軟正黑體', 14), variable=var, value=A, command=print_selection)
    r1.pack()
    r2 = tk.Radiobutton(window, text=B+' : '+Bp, font=('微軟正黑體', 14), fg='blue', variable=var, value=B, command=print_selection)
    r2.pack()
    r3 = tk.Radiobutton(window, text=C+' : '+Cp, font=('微軟正黑體', 14), fg='blue', variable=var, value=C, command=print_selection)
    r3.pack()
    b = tk.Button(window, text='確定', font=('微軟正黑體', 12), width=10, height=1, command=close)
    b.pack()
    
    window.mainloop()
    return var.get()
            
head_x = head_y = head_z = 0
body_x = body_y = body_z = 0
foot_x = foot_y = foot_z = 0
head = [0, 0]
body = [0, 0]
foot = [0, 0]
P = np.zeros((6, 15))
act = ['', '異常']
pose = np.array(['站', '走路', '坐下', '坐', '躺下', '躺', '起身', '起立', '跌倒', '彎腰', ''] )
poseindex = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
action = [[0, 1, 0], [0, 2, 3], [0, 1, 2], [0, 8, 5], [0, 1, 8], [1, 0, 1], [1, 0, 8], 
          [1, 0, 2], [1, 2, 3], [1, 8, 5], [2, 3, 4], [2, 3, 7], [2, 4, 5], [2, 7, 0],
          [2, 7, 1], [3, 4, 5], [3, 4, 6], [3, 7, 1], [3, 7, 0], [4, 5, 6], [4, 6, 3], 
          [4, 6, 7], [5, 6, 3], [5, 6, 7], [5, 6, 4], [6, 4, 5], [6, 3, 4], [6, 3, 7],
          [7, 0, 1], [7, 1, 0], [7, 0, 2], [7, 0, 8], [7, 1, 2], [7, 1, 8], [8, 5, 6]]
C = np.load('CP_1229.npy')
Pose = []
Action = np.empty((1,))
ActionBlock=[1000,1000,1000]
A = np.zeros((300,1))
n = 0
while True:
    s = time.time()  
    dframe = depth_stream.read_frame()
    frame = np.array(dframe.get_buffer_as_uint16()).reshape((60,80))
    n+=1 
    boxes,scores,labels=yolo.detect(frame)
    boxes = boxes.astype(np.int64)

    new_boxes = boxes 
    new_scores = scores
    new_labels = labels
    for i in range(len(boxes)):
        if scores[i] > 0.8:
            for j in range(len(boxes)):
                b = IOU(int(boxes[i,1]),int(boxes[i,0]),int(boxes[i,3]-boxes[i,1]),int(boxes[i,2]-boxes[i,0]),
                          int(boxes[j,1]),int(boxes[j,0]),int(boxes[j,3]-boxes[j,1]),int(boxes[j,2]-boxes[j,0]))  
                if b<1 and b>0.5:
                    new_boxes = np.delete(boxes, j, axis=0)  
                    new_scores = np.delete(scores, j, axis=0)
                    new_labels = np.delete(labels, j, axis=0)
            
    for i in range(len(new_boxes)):
        if new_scores[i] > 0.8:               
            box = new_boxes[i]         
            frame = cv2.rectangle(frame,(box[1],box[0]),(box[3],box[2]),0,1)
                        
            if new_labels[i] == 0 :
                head_x = int((box[3]+box[1])/2)
                head_y = int((box[2]+box[0])/2)               
                head_z = float(frame[head_y, head_x]/6000) 
                head = [float(head_x/80.), float(head_y/60.)]
                
            if new_labels[i] == 1 :
                body_x = int((box[3]+box[1])/2)
                body_y = int((box[2]+box[0])/2)
                body_z = float(frame[body_y, body_x]/6000)
                body = [float(body_x/80.), float(body_y/60.)]
                                
            if new_labels[i] == 2 :
                foot_x = int((box[3]+box[1])/2)
                foot_y = int((box[2]+box[0])/2)
                foot_z = float(frame[foot_y, foot_x]/6000)
                foot = [float(foot_x/80.), float(foot_y/60.)] 
                                        
    point = np.array(((head[0]),(head[1]),(body[0]),(body[1]),(foot[0]),(foot[1])))
    point = point.reshape(6, 1)      
    P = P[:,1:]
    P = np.concatenate((P, point), axis=1)  
    
    K = []   
    for _ in range(6):
        k = Kalman(P[_]).reshape(-1) 
        K.append(k) 
        w = []
        v = []   
        a = []
        if len(K)>4 :
            for j in range(14):
                w.append(toWorld(frame,[int(K[3][j]*60.),int(K[2][j]*80.)]))
                if len(w)>1 : v.append(w[-1][2] - w[-2][2]) 
                if len(v)>1 : a.append(v[-1] - v[-2])         
        
    K = np.array(K)       
    pred = model.predict(K[np.newaxis, :, :])   
    pred = pred.reshape(-1)

    #draw
    result = np.zeros(shape=(100, 120, 3), dtype = np.uint8)
    img = np.expand_dims(frame, axis=2)
    img = np.concatenate((img, img, img), axis=-1)
    result[20:80,20:100,:] = heatmap(img) 
    result = cv2.resize(result,(480,360))
    
    # debug
    if len(Pose)>1:       
        prepose = int(np.argwhere(pose==Pose[-2]))
        nowpose = int(np.argmax(pred))
        new_pred = pred*C[prepose]
    else:new_pred = pred
    Pose.append(pose[np.argmax(new_pred)])
    if len(Pose)<15:
        pose_result = pose[np.argmax(new_pred)]
    else:
        pose_result = get_mode(Pose[(n)-14:(n)+1])
    
    # fall detect
    for _ in range(12):
        if pose_result=='跌倒' and abs(a[_])<3000: 
            fall = Top(pred,2)
            nofall = '%s'%(pose[np.argwhere(fall[1])][0,0])
        elif pose_result=='跌倒' and abs(a[_])>=3000:
            result = cv2ImgAddText(result, '警告', 350, 300, (255, 0, 0), 60)
    
    # behavior analysis        
    num = np.array(poseindex[int(np.argwhere(pose==pose_result))]).reshape(1,)
    Action = np.concatenate((Action, num)) 
    if Action[0]==4e-322 : Action = np.delete(Action, 0, axis=0)
    if len(Action)>1 and Action[-1]==Action[-2] : Action = np.delete(Action, -2, axis=0)
          
    if ActionBlock[-1] != Action[-1] :
        ActionBlock = ActionBlock[1:]
        ActionBlock.append(Action[-1])                          
    Ab = np.array(ActionBlock)
    predb = modelb.predict(Ab[np.newaxis,np.newaxis, :]) 
   
#    if ActionBlock[0]!= 1000 and ActionBlock[0]!= 10 and act[np.argmax(predb)] == '異常':
    if ActionBlock[0]!= 1000 and ActionBlock[0]!= 10 and ActionBlock not in action:
        result = cv2ImgAddText(result, act[np.argmax(predb)], 350, 0, (255, 0, 0), 60)
        result = cv2ImgAddText(result, pose_result, 3, 300, (255, 255, 255), 60)
        
#        top = Top(C[int(Action[-2])],2)
#        pose_pre1 = '%s'%(pose[np.where(top[0]==C[int(Action[-2])])][0])
#        pose_pre2 = '%s'%(pose[np.where(top[1]==C[int(Action[-2])])][0])                                    
#        if c==2 :
#            var = UI(pose[int(Action[-2])],pose_result,pose_pre1,str(round(top[0], 2)),pose_pre2,str(round(top[1], 2)))           
#        if c>2 :
#            result = cv2ImgAddText(result, var, 3, 300, (255, 255, 255), 60)
#            Action[-1] = poseindex[np.where(top[0]==C[int(Action[-2])])]
#            if len(Action)>1 and Action[-1]==Action[-2] : Action = np.delete(Action, -2, axis=0)
#            ActionBlock = [Action[-3],Action[-2],Action[-1]]  
#            Ab = np.array(ActionBlock)
#            pose_result = pose[int(Action[-1])]                                                                                                                       
    else:       
        result = cv2ImgAddText(result, pose_result, 3, 300, (255, 255, 255), 60)                                
                                
    result = cv2ImgAddText(result, str(n), 3, 0, (255, 255, 255), 60)                   
    cv2.imshow("result", result)
#    print(time.time()-s)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


    