#2020.1.5 success on video detect
# need to check with oranges on tree
# ref: https://github.com/zhaoyanglijoey/yolov3
#Video python3 detector.py -i <input> -v
#Webcam python3 detector.py -v -w -i 0
import torch
import cv2
import numpy as np
import socket
import time
from torch.autograd import Variable
from darknet import Darknet
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
import sys
from datetime import datetime

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime



HOST = '192.168.43.179'
PORT = 8060 #21567 #8080 #8001
server_addr=(HOST,PORT)
global socket_cv
socket_cv=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def connect():
    HOST = '192.168.43.179'
    PORT = 8080 #21567 #8080 #8001
    server_addr=(HOST,PORT)
    socket_cv=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_cv.connect(server_addr)

def ClientSocket(flag,x,y,depth):
    command=np.array([flag,x,y,depth])
    for i in range(0,len(command)):
        c=str(command[i])
        com=c.encode()
        #l=str(len(command))
        socket_cv.send(com)
        r=socket_cv.recv(1024)
        #print(r)
        if r=='wrong':
            print('next')
            break
        while len(r)==0:
            time.sleep(100)
            r=socket_cv.recv(1024)
            #print(r)
            if r=='wrong':
                print('next')
                break
    #command=command.encode()
    #self.socket_cv.send(command)
    print('position passed')
    data1=socket_cv.recv(1024)
    data1=data1.decode()
    flag1=1
    print(data1)
    if data1=='wrong':
        print('next orange')
        flag1=0
    if data1=='cant reach':
        print("cant reach")
        flag1=0
    if data1=='finished':
        print('next orange')
        flag1=0
    time.sleep(1)
    while flag1:
        print('loop')
        time.sleep(1)
        data1=socket_cv.recv(1024)
        data1=data1.decode()
        print(data1)
        #different situation
        if data1=='finished':
            print("orange picked!")
            flag1=0
            break
        if data1=='cant reach':
            print("cant reach")
            flag1=0
            break
        if data1=='wrong':
            print('next')
            flag1=0
            break
    return flag1
    #if len(data1)>0:
    #    print("fruit picked!")



def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [imgs[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]

    return batches
#!!!for here return the center of the object
def draw_bbox(imgs, bbox, colors, classes):
    img = imgs[int(bbox[0])]
    label = classes[int(bbox[-1])]

    p1 = tuple(bbox[1:3].int())
    x1=p1[0].numpy()
    y1=p1[1].numpy()
    p2 = tuple(bbox[3:5].int())
    x2=p2[0].numpy()
    y2=p2[1].numpy()
    # x,y is the center of the bounding box
    x=(x1+x2)/2
    y=(y1+y2)/2

    #save as tuple
    center=(int(x),int(y))
    #2020.1.15

    color = random.choice(colors)
    
    #bouding box
    cv2.rectangle(img, p1, p2, color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    # text background
    cv2.rectangle(img, p3, p4, color, -1)
    #center
    cv2.circle(img,center,5,color,thickness=-1)
    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)
 
    return center


def get_depth(point,kinect):
    frame_depth=kinect._depth_frame_data
    x=point[0]
    y=point[1]
    k=0.105
    nx=int(x+x*k)   
    pixel_depth=frame_depth[(((22+y)*512)+nx)]
    
    return nx,x,y,pixel_depth



def detect_video(model):
    kinect=PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    colors = pkl.load(open("yolov3-master\pallete", "rb"))
    classes = load_classes("yolov3-master\data\coco.names")
    colors = [colors[1]]

    # cap is the video captured by camera 
    # for surface 0 is front 1 is back 2 is kinect
    cap = cv2.VideoCapture(1+cv2.CAP_DSHOW)

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    
    read_frames = 0

    start_time = datetime.now()
    #print('Start Detect')
    #set times as 1
    #num=0
    #sample array
     #s=[[] for i in range(20)]
    #s=[]
    #!while loop!
    locationx=[]
    locationy=[]
    locationd=[]
    while cap.isOpened():

        #print('Detecting')
        retflag, frame = cap.read()
        '''
        frame need to be corpped and then resize to 512*424
        '''
        
        cv2.circle(frame,(960,540),5,[0,255,255],thickness=-2)
        #2 feet-41
        #3 feet-41
        #4 feet-37
        #x=154
        y=0
        h=1080
        #h=1272
        #w=1611
        w=1920/84.1*70.6
        w=int(w)
        x=(1920-w)/2+140
        x=int(x)
        dim = (512, 380)
        frame = frame[y:y+h, x:x+w]
        frame=cv2.resize(frame,dim)

        read_frames += 1
        if retflag:
            '''
            get depth frame
            '''
            if kinect.has_new_depth_frame():
                Dframe = kinect.get_last_depth_frame()
                frameD = kinect._depth_frame_data
                Dframe = Dframe.astype(np.uint8)
                #print(frame)
                
                Dframe = np.reshape(Dframe,(424,512))
                dx=0
                dy=22
                dh=380                
                dw=512
                dim = (512, 380)
                Dframe = Dframe[dy:dy+dh, dx:dx+dw]
                frame=cv2.resize(frame,dim)

                Dframe = cv2.cvtColor(Dframe,cv2.COLOR_GRAY2RGB)
                def click_event(event,x,y,flags,param):
                    if event == cv2.EVENT_RBUTTONDOWN:
                        print(x,y)
                    if event == cv2.EVENT_LBUTTONDOWN:
                        Pixel_Depth = frameD[(((22+y)*512)+x)]
                        print("x ",x,"y ",y,"Depth",Pixel_Depth)
                
                
                '''
                get RGB frame
                '''
            frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            #if torch.cuda.is_available:
            frame_tensor = frame_tensor.cuda()
            detections = model(frame_tensor,True).cpu()
            #orange order
            flag0=0
            flag1=1
            detections = process_result(detections, 0.5, 0.4)
            if len(detections) != 0:
                #3.3
                flag=0
                detections = transform_result(detections, [frame], input_size)
                num=len(detections)
        
                for detection in detections:
                    Label=int(detection[-1])
                    flag=flag+1
                    if Label == 49:
                        flag=flag+1
                        
                        #detection=[]
                        center=draw_bbox([frame], detection, colors, classes)
                    #print(Label)
                    #print('cc is',center)
                        Dcenter=draw_bbox([Dframe],detection,colors,classes)
                        #import k
                        #k=0.105
                        #x,y,d=get_depth(center,kinect,k)
                        
                        #redraw the boundary box
                        img = Dframe
                        nx,x,y,d=get_depth(center,kinect)                        
                        cv2.circle(img,(nx,y),5,[0,0,255],thickness=-1)
                        #send to robot
                        x=center[0]
                        y=center[1]
                        # change point into the camera center
                        x=x-512/2+40
                        y=y-318/2-35
                        y=-y
                        x=-x
                        #position
                        if flag==3:
                            print("x ",x,"y ",y,"d ",d)
                            l=len(locationx)
                        
                        #if True:
                        #if flag==flag1:
                            #approciate depth
                            if d>500 and d<1050:
                                locationx.append(x)
                                locationy.append(y)
                                locationd.append(d)
                                print(locationx)
                                print(l)
                                if l>6:
                                    x1=locationx[l-6:l-1]
                                    y1=locationy[l-6:l-1]
                                    d1=locationd[l-6:l-1]
                                    diff=np.var(x1)+np.var(y1)+np.var(d1)
                                    print(x1)
                                    print(y1)
                                    print(d1)
                                    print(diff)
                                    # less fluctuation
                                    if diff<20:
                                        print("get position")
                                        x=np.average(x1)
                                        y=np.average(y1)
                                        d=np.average(d1)
                                        #scale x,y
                                        k1=1.48
                                        x=x*k1*d/512
                                        y=y*k1*d/318*0.72
                                        #x=x*k1
                                        #y=y*k1
                                        #actual position
                                        print(flag0,x,y,d)
                                        #send to raspbaerry pi
                                        flag2=ClientSocket(flag0,x,y,d)
                                        #reset the data
                                        flag0=1
                                        locationx=[]
                                        locationy=[]
                                        locationd=[]
                                        

                        #choose stable on as result
                        #sample function()
                        #num=num+1
                        

                        
            cv2.imshow('RGBFrame',frame)
            cv2.imshow('DepthFrame',Dframe)
          
            #print("x: ", x ,"y: ", y)
            #print("_______________________")
           
            if read_frames % 60 == 0:
                #check the center
                locationx=[]
                locationy=[]
                locationd=[]
                center11=[211,190]
                nx11,x11,y11,d11=get_depth(center11,kinect)
                print(d11)
                #print('Number of frames processed:', read_frames)
                
                #print('average FPS',float(read_frames/datetime.now()))
            if flag0:
                locationx=[]
                locationy=[]
                locationd=[]

            if not False and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)
    cap.release()

    cv2.destroyAllWindows()

    return

def detect(model):
    detect_video(model)

def main():


    model = Darknet('yolov3-master\cfg\yolov3.cfg')
    model.load_weights('model\yolov3.weights')
    if torch.cuda.is_available():
        model.cuda()
    socket_cv.connect(server_addr)

    #while True:
    #print("Connecting tp server @%s:%d..." %(HOST,PORT)) 
    print('Loading network...')
    model.eval()
    print('Network loaded')
    detect(model)
   
    #detect_video(model)



if __name__ == '__main__':
    main()