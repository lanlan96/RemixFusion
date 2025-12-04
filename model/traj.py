import numpy as np 
import math
import time
import os

from model.utils import matrixtoeular


class traj:
    def __init__(self,time_stamp,tx,ty,tz,qx,qy,qz,qw,RT):
        # print()
        self.timestamp=time_stamp
        self.tx=tx
        self.ty=ty
        self.tz=tz
        normal=math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
        self.qx=qx/normal
        self.qy=qy/normal
        self.qz=qz/normal
        self.qw=qw/normal
        self.RT=RT

class Trajectory:
    def __init__(self,dir_path):#,fixed):
        self.path=os.path.join(dir_path,"trajectory.txt")
        self.pose_list=[]
        #added
        self.kfx=0.0
        self.kfy=0.0
        self.kfz=0.0
        self.first = 0
   

        self.cam_vec=np.asarray([0,0,0]).astype(np.float32)
        self.first_axis="x"

    def transform_to_quaternion(self,R):
        tr=R[0,0]+R[1,1]+R[2,2]
        if (tr>0):
            S=math.sqrt(tr+1)*2
            qw=0.25*S
            qx=(R[2,1]-R[1,2])/S
            qy=(R[0,2]-R[2,0])/S
            qz=(R[1,0]-R[0,1])/S
            return qx,qy,qz,qw
        elif ((R[0,0]>R[1,1] and R[0,0]>R[2,2])):
            S=math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
            qw=(R[2,1]-R[1,2])/S
            qx=0.25*S
            qy=(R[0,1]+R[1,0])/S
            qz=(R[0,2]+R[2,0])/S
            return qx,qy,qz,qw
        elif (R[1,1]>R[2,2]):
            S=math.sqrt(1+R[1,1]-R[0,0]-R[2,2,])*2
            qw=(R[0,2]-R[2,0])/S
            qx=(R[0,1]+R[1,0])/S
            qy=0.25*S
            qz=(R[1,2]+R[2,1])/S
            return qx,qy,qz,qw
        else:
            S=math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
            qw=(R[1,0]-R[0,1])/S
            qx=(R[0,2]+R[2,0])/S
            qy=(R[1,2]+R[2,1])/S
            qz=0.25*S
            return qx,qy,qz,qw

    def add_trajectory_all(self,timestamp,transform):
        qx,qy,qz,qw=self.transform_to_quaternion(transform[:3,:3])
        self.pose_list.append(traj(timestamp,
                                        transform[0,3],
                                        transform[1,3],
                                        transform[2,3],
                                        qx,
                                        qy,
                                        qz,
                                        qw,
                                        transform))

    def write_trajectory(self):
        tra_file=open(self.path,"w")
        for tra in self.pose_list:
            tra_file.write("%f %f %f %f %f %f %f %f\n"%(                        
                        float(tra.timestamp),
                        tra.tx,
                        tra.ty,
                        tra.tz,
                        tra.qx,
                        tra.qy,
                        tra.qz,
                        tra.qw))
            
    def write_trajectory_RT(self):
        tra_file=open(self.path,"w")
        for tra in self.pose_list:
            tra_file.write("%f %f %f %f %f %f %f %f %f %f %f %f\n"%(                        
                        tra.RT[0,0],
                        tra.RT[0,1],
                        tra.RT[0,2],
                        tra.RT[0,3],
                        tra.RT[1,0],
                        tra.RT[1,1],
                        tra.RT[1,2],
                        tra.RT[1,3],
                        tra.RT[2,0],
                        tra.RT[2,1],
                        tra.RT[2,2],
                        tra.RT[2,3]))
    