#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import pdb
import os.path
import time
import datetime
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError
from rosgraph_msgs.msg import Clock




class BagVideoSynchronizer():

    def __init__(self, filename_list, time_diff=0, slack=100):

        self.filename_list = filename_list

        #information from the files
        self.start_time_arr = None
        self.end_time_arr = None
        self.cur_time_arr = None
        self.cap_list = []
        self.sync_index = 0

        self.resize_dim = None

        self.time_diff = time_diff
        self.slack = slack

        self.initialize_information()
        for i in range(len(self.cap_list)):
            print 'Start time :', time.ctime(self.start_time_arr[i])
            print 'End time :', time.ctime(self.end_time_arr[i])

        self.pubimg = rospy.Publisher('video_frame', Image, queue_size=10)
        self.pubts = rospy.Publisher('video_timestamp', Float64, queue_size=10)
        self.bag_sub = rospy.Subscriber('/clock', Clock, self.synchronizer)

        self.final_time = np.max(self.end_time_arr)
        self.bridge = CvBridge()


    def initialize_information(self):

        scale_percent = 40
        start_time_list = []
        end_time_list = []
        cur_time_list = []
        cap_list = []
        for file in self.filename_list:

            cap = cv2.VideoCapture(file)
            cap_list.append(cap)
            end_time = os.path.getmtime(file) + self.time_diff
            end_time_list.append(end_time)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            start_time = end_time - (total_frames/fps)
            cur_time = start_time
            start_time_list.append(start_time)
            cur_time_list.append(cur_time)
            width = cv2.CAP_PROP_FRAME_WIDTH * scale_percent
            height = cv2.CAP_PROP_FRAME_HEIGHT * scale_percent
            resize_dim = (width, height) 


        self.cap_list = cap_list
        self.start_time_arr = np.asarray(start_time_list)
        self.end_time_arr = np.asarray(end_time_list)
        self.cur_time_arr = np.asarray(cur_time_list)
        self.resize_dim = resize_dim


    def synchronizer(self, data):

        bag_time = data.clock.secs + float(data.clock.nsecs)/10**9

        print "bag_time :", time.ctime(bag_time)
        diff = np.abs(self.cur_time_arr - bag_time)

        for i in range(len(self.cap_list)):
            print 'times in video :', time.ctime(self.cur_time_arr[i]), ' : ', diff[i]

        index = np.argmin(diff)

        print '++++++++++++++++++++++'

        if diff[index] < self.slack:

            cap = self.cap_list[index]
            ret, frame = cap.read()

            if frame is not None:

                #frame = cv2.resize(frame, self.resize_dim)
                self.cur_time_arr[index] = self.start_time_arr[index] + cap.get(cv2.CAP_PROP_POS_MSEC)/float(1000)
                image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                print self.cur_time_arr[index]
                self.pubimg.publish(image_msg)
                self.pubts.publish(self.cur_time_arr[index])


        if bag_time > self.final_time:

            for cap in self.cap_list:

                cap.release()
            
            cv2.destroyAllWindows()








'''
def publish_image_time_stamp(filenames, slack=0):

    files = filenames

    cap_list = []
    start_time_list = []
    end_time_list = []
    for file in files:
    cap = cv2.VideoCapture(file)
    end_time = os.path.getmtime(file)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = end_time - (total_frames/fps) + slack
    cur_time = start_time



    bridge = CvBridge()

    while(cur_time < end_time):
        ret, frame = cap.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale_percent = 30 # percent of original size
        if frame is not None:
            
            frame = cv2.resize(frame, dim)
            cur_time = start_time + cap.get(cv2.CAP_PROP_POS_MSEC)/float(1000)
            image_msg = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
            pubimg.publish(image_msg)
            pubts.publish(cur_time)

    cap.release()
    cv2.destroyAllWindows()
'''

if __name__ == '__main__':

    rospy.init_node('video_info_publisher')
    '''
    time_diff for gopro 2 = -3596
    time_diff for gopro 1 = -3568 or -3567
    time_diff for gopro 3 = 1987
    '''
    filename_list = []
    folder_name = '/media/abhisek/Backup Plus/bbds2020/monday/gopro3/monday'


    for r, d, f in os.walk(folder_name):
        for file in f:
            if '.MP4' in file:
                filename_list.append(os.path.join(r, file))

    print filename_list
       
    bvs = BagVideoSynchronizer(filename_list, time_diff = -3450, slack=0.3)

    rospy.spin()