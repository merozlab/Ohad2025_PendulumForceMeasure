# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:19:53 2025

functions for exp2 - Pendulum force

A. plant class
B. event class
0. get tracked data
1. calculate angle relative to vertical in side and top views
2. calc angle for time series
3. calculate force in mN of bean on support via moment equilibrium equation,
    in units of cgs.
4. calculate force for time series

@author: Amir
"""
#%% Imports
import math as m
import numpy as np
import re
import time
import seaborn
import scipy
import sys
import os
import h5py
from scipy.signal import savgol_filter,resample
import os,glob
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

sys.path.append('..')
import useful_functions as uf # my functions
#%% A - Plant class
class Plant:
    """ plant class, insert all parameters from XL, Youngs modulus,
        top and side trajectories"""

    def __init__(self,df,basepath,i,exp):
        self.exp_num = exp
        self.plant_path = os.path.join(basepath,r'Measurements',
           str(self.exp_num),r'Side',r'cn')

        self.genus = df.at[i,'Bean_Strain']
        self.camera = df.at[i,'Camera']  # images from nikon or pi
        self.m_sup = float(df.at[i,'Straw_Weight(gr)']) # support mass

    def view_data(self,df_tot,i):
        df = df_tot.iloc[i]
        self.pix2cm_s = float(df.at['Side_pix2cm']) # side pixel to cm ratio
        self.pix2cm_t = float(df.at['Top_pix2cm']) # top pixel to cm ratio

        # pendulum lenth in cm: add measured support length with pix2cm converted straw2hinge 
        self.Lsup_cm = df.at['Dist_straw_from_hinge(pixels)']*self.pix2cm_s + df.at['Straw_Length(cm)'] 
        # self.Lsup_cm = self.Lsup_pix*self.pix2cm_s # support lenth in cm

        # z position of bottom tip of support
        self.support_base_z_pos_pix = float(df.at['side_equil_ypos-bot_sup(pixels)'])
        try: # if new measurement exists
            self.support_base_z_pos_pix_new = float(df.at['new_y_pos_supp_bot(pixels)']) # updated z position from side image of supp bottom
        except: 
            self.support_base_z_pos_pix_new = self.support_base_z_pos_pix
            print('no new z position of support bottom')
        # self.support_base_z_pos_cm = self.support_base_z_pos_pix*self.pix2cm_s # not used


    def cn_data(self,df,i):
        '''df=data drame, i=index of data frame'''
        self.T,self.avgT = uf.get_Tcn(self.plant_path,df,i) # get Tcn from excel or folder
        self.omega0 = 2*m.pi/self.avgT # base rotation angular velocity

#%% B - Event class
class Event:
    """event class, plant as input"""
    def __init__(self,plant,df,i):
        self.p = plant

    def view_data(self,df,i,view):
        if view == 'side':
            self.frm0_side = int(df.at[i,'First_contact_frame']) # first contact frame (starts from 1)
            self.frm_dec_side = int(df.at[i,'Slip/Twine_frame']) # frame of twine_state/slip
            self.event_label = df.at[i,'Event Label'] # event label

        if view == 'top':
            self.frm0_top = int(df.at[i,'First_contact_frame']) # first contact frame (starts from 1)
            self.frm_dec_top = int(df.at[i,'Slip/Twine_frame']) # frame of twine_state/slip
            self.L_contact2stemtip_cm = self.p.pix2cm_t * float(df.at[i,
                'Contact_distance_from_stem_tip(pixels)']) # contact distance from stem tip at initial contact time
            self.twine_state = float(df.at[i,'Twine_status']) # twine_state/slip


    def event_base_calcs(self,view,track_dict,contact_dict):
        '''side view:
        -x contact pix, x,z contact cm,
        -distance of track position to support tip: h_tip (L_track2suptip)
        -track timer, contact timer


        top view:
        -get coordinates of x,y track pix,
        -x,y track cm. 
        -get x,y relative to x0,y0 (relative to average before contact?)
        -extract phi=atan(y/x)- check direction...
        -get size of r_tr=x/cos(phi)
        -get size of r_tr=sqrt(x^2+y^2) - compare?
        -get alpha=asin(r_tr/(2*(L-h_tip))
        side view:
        - get coordinates of x,z track pix
        - get coordinates of x,z contact pix
        do not rely on pix2cm conversion of these values. 
        if i just use the pix to get the angle, i can use known length of the section to get the wanted length?

        combine top and side view:
        -get r_co=x_c0/cos(phi)
        -get l_c0 = r_co/sin(alpha)'''

        if view == 'side': # if side view
            # x,z coordinates of side view tip tracking
            self.x_track_side0,self.z_track_side0,time_s = funcget_tracked_data(
                track_dict[(self.p.exp_num,self.event_num,view)][0]
                ,[0,-1],view,self.p.camera)
            
            # x0,z0 - side view equilibrium coordinates of tracked point
            self.x0_side,self.z0_side = self.x_track_side0[0],self.z_track_side0[0]

            # transform x,z to coordinates relative to x0,z0
            self.x_track_side,self.z_track_side = \
                -np.subtract(self.x_track_side0,self.x0_side),\
                np.subtract(self.z_track_side0,self.z0_side)
            
            # convert to cm
            self.x_track_side_cm = np.multiply(self.x_track_side,self.p.pix2cm_s)
            self.z_track_side_cm = np.multiply(self.z_track_side,self.p.pix2cm_s)

            # save within decision timeframe
            self.x_track_side_dec = self.x_track_side_cm[self.frm0_side:self.frm_dec_side]
            self.z_track_side_dec = self.z_track_side_cm[self.frm0_side:self.frm_dec_side]

            # get contact data from side view
            self.x_cont,self.z_cont,self.contact_timer = funcget_tracked_data(
                contact_dict[(self.p.exp_num,self.event_num)][0],[0,-1],
                view,self.p.camera,1)
            # transform contact x,z to coordinates relative to x0,z0
            # z_cont should always be larger than z0
            # compare x contact to the initial x position of the tracked contact point - not the support tip
            # take minus the x since the direction is opposite to the top view
            self.x_cont,self.z_cont = \
                -np.subtract(self.x_cont,self.x_cont[0]),abs(np.subtract(self.z_cont,self.z0_side))
            # convert to cm
            self.x_cont_cm = np.multiply(self.x_cont,self.p.pix2cm_s)
            self.z_cont_cm = np.multiply(self.z_cont,self.p.pix2cm_s)

            # save to dict only coor within decision timeframe
            # self.xz_contact = np.array([self.x_cont_cm[self.frm0_side:self.frm_dec_side],
            #                 self.z_cont_cm[self.frm0_side:self.frm_dec_side]])
            self.x_cont_dec = self.x_cont_cm[self.frm0_side:self.frm_dec_side]          
            self.z_cont_dec = self.z_cont_cm[self.frm0_side:self.frm_dec_side]

            # get z dist. of tracked spot at equil. to bottom of support + convert to cm
            self.L_track2suptip = abs(self.z0_side-self.p.support_base_z_pos_pix)
            self.L_track2suptip_new = abs(self.z0_side-self.p.support_base_z_pos_pix_new) # updated track2tip pixels
            self.L_track2suptip_cm = self.L_track2suptip*self.p.pix2cm_s
            self.L_track2suptip_cm_new = self.L_track2suptip_new*self.p.pix2cm_s # updated track2tip in cm
            self.h_tip = self.L_track2suptip_cm
            self.h_tip_new = self.L_track2suptip_cm_new # updated 
            self.L_tracked = self.p.Lsup_cm - self.h_tip # length of support from hinge to tracked point in cm
            self.L_tracked_new = self.p.Lsup_cm - self.h_tip_new # updated length of support from hinge to tracked point in cm


        else: # if top view
            self.x_track_top_pix,self.y_track_top_pix,self.top_timer = funcget_tracked_data(
                track_dict[(self.p.exp_num,self.event_num,view)][0]
                ,[0,-1],view,self.p.camera)
            
            # x0,y0 - top view equilibrium coordinates of tracked point
            self.x0_top,self.y0_top = self.x_track_top_pix[0],self.y_track_top_pix[0]

            # transform x,y to coordinates relative to x0,y0 in top view
            self.x_track_top,self.y_track_top = \
                np.subtract(self.x_track_top_pix,self.x0_top),np.subtract(self.y_track_top_pix,self.y0_top)
            
            # convert to cm
            self.x_track_top_cm = np.multiply(self.x_track_top,self.p.pix2cm_t)
            self.y_track_top_cm = np.multiply(self.y_track_top,self.p.pix2cm_t)

            # set length of events to dec period only
            self.dec_x_track_top = self.x_track_top_cm[self.frm0_top:self.frm_dec_top]
            self.dec_y_track_top = self.y_track_top_cm[self.frm0_top:self.frm_dec_top]
            self.dec_z_track_side = self.z_track_side_cm[self.frm0_side:self.frm_dec_side]

            # timer for decision period
            self.timer = np.subtract(self.top_timer[self.frm0_top:
                        self.frm_dec_top],self.top_timer[self.frm0_top])
            
            # collect track xyz (in cm) - distance from eq point of tracked point in cm
            self.xyz = np.zeros((3,len(self.dec_x_track_top))) # all xyz of sup track
            self.xyz[0,::] = self.dec_x_track_top
            self.xyz[1,::] = self.dec_y_track_top
            self.dec_z_track_side,self.dec_y_track_top = \
                uf.adjust_len(self.dec_z_track_side,self.dec_y_track_top,
                  choose=self.dec_y_track_top)
            self.xyz[2,::] = self.dec_z_track_side

            # self.xyz = np.array(self.xyz_supportrack) # in cm ?
            self.xyz0 = np.array([[self.xyz[0][0],self.xyz[1][0],
                self.xyz[2][0]]]*len(self.xyz[0])).T # initial xyz trk point

    def event_calc_variables(self,view):
        if view == 'side':
            return
        else:
            # adjust coordantes lengths
            self.z_cont_dec,self.xyz[0] = uf.adjust_len(self.z_cont_dec,self.xyz[0],choose=self.xyz[0])

            ###################################################################
            # 3rd calculation: get support vector

            # set equilibrium point as origin (0,0,0), and support hinge as (0,0,L_tracked) in cm
            # self.hinge = np.array([0,0,self.L_tracked])
            self.hinge_xyz = np.array([0,0,self.L_tracked_new]) # zsup updated, hinge_updt_zsup

            # get r_tr from sum of x and y components squared
            self.r_tr_xy_raw = np.sqrt(self.xyz[0]**2 + self.xyz[1]**2)
            # self.r_tr_xy = self.xyz[0]**2 + self.xyz[1]**2 # check without sqrt
            self.r_tr_xy = self.r_tr_xy_raw - self.r_tr_xy_raw[0] # relative to start point

            # calculate alpha via asin(r_tr/((L-h_tip)))
            # self.alpha3 = np.arcsin(np.divide(self.r_tr_xy,self.L_tracked))
            self.alpha = np.arcsin(np.divide(self.r_tr_xy,self.L_tracked_new)) # zsup updated,alpha3_updt_zsup

            # p*(x,y,z)track is then the parametrized vector describing the support
            # transpose to allow subtraction of hinge from each point, then transpose back
            # self.dxyz = np.subtract(self.xyz.T,self.hinge).T # vector from hinge to track point
            self.dxyz = np.subtract(self.xyz.T,self.hinge_xyz).T # zsup updated,dxyz_updt_zsup

            # self.dz_cont = np.subtract(self.z_cont_dec,self.hinge[2]) # z distance from hinge to contact point
            self.dz_contact = np.subtract(self.z_cont_dec,self.hinge_xyz[2]) # zsup updated, dz_contact_updt_zsup

            # we extract p for the contact point from the relation
            # p = xc/xtr = yc/ytr = zc/ztr. (but we dont know y from this description)
            # self.px = np.divide(self.x_cont_dec,self.dxyz[0][:]) # denominator will be close to zero...
            # self.px_side = np.divide(self.x_cont_dec,self.x_track_side_dec) # using side view coordiantes only

            # self.pz = np.divide(self.dz_contact,self.dxyz[2][:]) # denominator wont be zero, since dz is ~ L_tracked
            self.pz = np.divide(self.dz_contact,self.dxyz[2][:]) # zsup updated,pz_updt_zsup

            # get py from the less volitile p (being pz):
            # self.yc = np.multiply(self.pz,self.dxyz[1][:])
            self.yc = np.multiply(self.pz,self.dxyz[1][:]) # zsup updated,yc_updt_zsup

            # then we get the contact position (x,y,z)c = p*(xtr,ytr,ztr)
            # self.xyz_contact = np.array(self.xz_contact[0],self.yc,self.xz_contact[1])
            # self.xyz_contact = np.multiply(self.pz,self.dxyz)
            self.xyz_contact = np.multiply(self.pz,self.dxyz) # zsup updated,xyz_contact_updt_zsup

            # so the contact length lc = sqrt(xc^2+yc^2+zc^2)
            # self.l_c_vec = np.sqrt(np.sum(self.xyz_contact**2,axis=0))
            self.l_contact = np.sqrt(np.sum(self.xyz_contact**2,axis=0)) # zsup updated, l_c_vec_updt_zsup

            # calculate force 3 - use same alpha as in method 2
            # self.F_bean_3 = F_of_t(self.l_c_vec,
            #           self.p.Lsup_cm, self.alpha3,self.p.m_sup,F_method=2)
            self.F_bean = F_of_t(self.l_contact,
                      self.p.Lsup_cm, self.alpha,self.p.m_sup,F_method=2) # zsup updated,F_bean_3_updt_zsup


#%% 0. Get tracked data
def funcget_tracked_data(filename,obj=0,view=[],camera='nikon',contact=[]):
    with open(filename,"r") as datafile:
        lines= datafile.readlines()
        # del lines[0] # remove first line to avoid 2 zero times
        N=np.size(lines,0) # number of lines
        xtl=[[]]*N # x top left
        ytl=[[]]*N # y top left
        w=[[]]*N # box width
        h=[[]]*N # box height
        index=[[]]*N # tracked object index
        xcntr=[[]]*N # box x center
        ycntr=[[]]*N # box y center
        # dist=[[]]*N # distance from equilibrium position
        timer=[[]]*N # time marks
        timer[0]=0 # time starts at zero
        # timer_epoch1=[[]]*N
        i=0 # count rows
# x,y,w,h ; start at upper left corner
        for line in lines:
            if line==[]: break
            currentline = line.split(",") # split by ','
            index[i]=int(currentline[-2]) # get index of tracked object
            if index[i] in obj:        # if current line belongs to the requested tracked object
                xtl[i]=float(currentline[0]) # xtl- x top left
                ytl[i]=float(currentline[1])
                w[i]=float(currentline[2])
                h[i]=float(currentline[3])
                xcntr[i]=xtl[i]+w[i]/2 # calculate x coordinate of box center
                ycntr[i]=ytl[i]-h[i]/2
                # if view=='top':
                #     dist[i]=np.sqrt((xcntr[i]-xcntr[0])**2+(ycntr[i]-ycntr[0])**2)
                # else:
                #     dist[i]=abs(xcntr[i]-xcntr[0])
                if camera=='nikon':
                    timer = [30*x for x in range(N)]

            else:
                print('skipped non-selected tracked object')
                print(timer[i],type(timer[i]),i,currentline)
            i+=1

        # x = np.subtract(xcntr,xcntr[0]) # return x relative to start point
        # y = np.subtract(ycntr,ycntr[0]) # return y relative to start point
        return xcntr,ycntr,timer
#%% 1. Calculate angle relative to vertical in side and top views
def calc_angle(lsup_pix,lsup_cm,dist_pix,pix2cm,view):
    if view=='side':
        alpha_deg=m.asin(dist_pix/lsup_pix) #pix calculation
    elif view=='top':
        alpha_deg=m.asin((dist_pix*pix2cm)/lsup_cm) #cm calculation
    return alpha_deg
# notice that if im in top view i dont have the lpix from the image, only the
# actual size from the side view/direct measurement
#%% 2. Calculate angle for time series
def alpha_of_t(lsup_pix,lsup_cm,dist_pix,pix2cm,view):
    # dlsup_pix,dlsup_cm,ddist_pix
    N=np.size(dist_pix) #size of distance vector
    angle=[[]]*N
    dangle=[[]]*N
    for i in range(N):
        angle[i]=calc_angle(lsup_pix,lsup_cm,dist_pix[i],pix2cm,view)
        # dangle[i]=calc_dangle(lsup_pix,dlsup_pix,lsup_cm,dlsup_cm,dist_pix[i],ddist_pix,pix2cm,view)
    return angle,dangle

#%% 3. Calculate force in mN

def calc_F_2(l_c,l_sup_cm,alpha_t,m_sup):
    '''calc using distance of contact from support hinge - l_c'''
    gcgs=980
    dyne2mN = 1/100
    F_mN = gcgs * m_sup * l_sup_cm * (m.tan(alpha_t)/(2*l_c)) * dyne2mN
    return abs(F_mN)
#%% 4. Calculate force for time series
def F_of_t(d_contact,l_sup_cm,alpha,m_sup,F_method=1):
    N=np.size(alpha)
    Fvec=[[]]*N
    if F_method==1:
        for i in range(N):
            Fvec[i] = calc_F_1(d_contact[i], l_sup_cm, alpha[i], m_sup)
    elif F_method==2:
        for i in range(N):
            Fvec[i] = calc_F_2(d_contact[i], l_sup_cm, alpha[i], m_sup)
    return Fvec 
#%%
