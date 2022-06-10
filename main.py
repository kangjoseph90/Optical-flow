from copy import deepcopy
from re import S
import pygame as pg
import pygame.camera as camera
from pygame.locals import *
import os,sys

import matplotlib.pyplot as plt

import numpy as np
import cv2

pg.init()
camera.init()

size=[1280,720]
screen=pg.display.set_mode(size)

clock=pg.time.Clock()

cam=camera.Camera(camera.list_cameras()[0],(320,180))
cam.start()

point=np.array([100,100],dtype=np.float32)
scale=2
    
grid=10

kernel1d=cv2.getGaussianKernel(2*grid-1,20)
kernel=np.outer(kernel1d,kernel1d.transpose())

def horn_shunck(rt,rx,ry,sx,sy):
    alpha=2
    
    u=np.zeros((sy+2,sx+2))
    v=np.zeros((sy+2,sx+2))
    
    nu=np.zeros((sy+2,sx+2))
    nv=np.zeros((sy+2,sx+2))
    
    iteration=10
    
    for _ in range(iteration):
        for i in range(sx):
            for j in range(sy):
                pixel=((j+2)*grid,(i+2)*grid)
                du=(u[j,i+1]+u[j+2,i+1]+u[j+1,i]+u[j+1,i+2])/4
                dv=(v[j,i+1]+v[j+2,i+1]+v[j+1,i]+v[j+1,i+2])/4
                nu[j+1,i+1]=du-rx[pixel]*(rx[pixel]*du+ry[pixel]*dv+rt[pixel])/(4*alpha**2+rx[pixel]**2+ry[pixel]**2)
                nv[j+1,i+1]=dv-ry[pixel]*(rx[pixel]*du+ry[pixel]*dv+rt[pixel])/(4*alpha**2+rx[pixel]**2+ry[pixel]**2)
        u,v=nu,nv
        
    return u,v


def lukas_kanade(x,y,rt,rx,ry):

    x=round(x)
    y=round(y)

    nrt=rt[y+1:y+2*grid,x+1:x+2*grid]
    nrx=rx[y+1:y+2*grid,x+1:x+2*grid]
    nry=ry[y+1:y+2*grid,x+1:x+2*grid]

    a1=np.array([[np.sum(nrx*nrx*kernel),np.sum(nrx*nry*kernel)],[np.sum(nrx*nry*kernel),np.sum(nry*nry*kernel)]])
    a2=np.array([[np.sum(nrx*nrt*kernel)],[np.sum(nry*nrt*kernel)]])

    v=np.zeros((2,1))

    if abs(np.linalg.det(a1)) <1: 
        return np.zeros((2,1))
    return -np.linalg.solve(a1,a2)


last_imgdata=pg.surfarray.array3d(cam.get_image()).swapaxes(0,1).astype(np.float32)
last_imgdata=cv2.cvtColor(last_imgdata,cv2.COLOR_BGR2GRAY)

last_imgdata=np.pad(last_imgdata,((grid,grid),(grid,grid)),'constant',constant_values=0)

while True:

    dt=clock.tick(24)/1000
    screen.fill((0,0,0))
    frame=cam.get_image()
    frame_resize=pg.transform.scale(frame,scale*np.array(frame.get_size()))
    screen.blit(frame_resize,(0,0))
    
    imgdata=pg.surfarray.array3d(frame).swapaxes(0,1).astype(np.float32)
    imgdata=cv2.cvtColor(imgdata,cv2.COLOR_BGR2GRAY)  #240,320
    imgdata=np.pad(imgdata,((grid,grid),(grid,grid)),'constant',constant_values=0)
    imgdata=cv2.medianBlur(imgdata,3)
  
    
    rt=(imgdata-last_imgdata)/dt
    rx=(np.roll(last_imgdata,-1,axis=1)-np.roll(last_imgdata,1,axis=1))/2
    ry=(np.roll(last_imgdata,-1,axis=0)-np.roll(last_imgdata,1,axis=0))/2

    """
    for y in range(grid,imgdata.shape[0]-3*grid+1,grid):
        for x in range(grid,imgdata.shape[1]-3*grid+1,grid):

            v=lukas_kanade(x,y,rt,rx,ry)

            pg.draw.line(screen,(255,0,0),[scale*x,scale*y],[scale*x+v[0,0]/3,scale*y+v[1,0]/3],2)
            pg.draw.circle(screen,(0,255,0),[scale*x,scale*y],2)
    """
    
    sx,sy=int(imgdata.shape[1]/grid)-3,int(imgdata.shape[0]/grid)-3
    u,v=horn_shunck(rt,rx,ry,sx,sy)
    
    for i in range(sx):
        for j in range(sy):
            pixel=((j+1)*grid,(i+1)*grid)

            pg.draw.line(screen,(255,0,0),[scale*pixel[1],scale*pixel[0]],[scale*pixel[1]+u[j+1,i+1]/3,scale*pixel[0]+v[j+1,i+1]/3],2)
            pg.draw.circle(screen,(0,255,0),[scale*pixel[1],scale*pixel[0]],2)
            
    
    
    temp=deepcopy(point)
    point+=lukas_kanade(point[0],point[1],rt,rx,ry)[:,0]*dt*1.5
    if point[0]-grid<0 or point[0]+grid>imgdata.shape[1] or point[1]-grid<0 or point[1]+grid>imgdata.shape[0]:
        point=temp


    pg.draw.circle(screen,(0,0,255),(point*scale).astype(int),5)

    last_imgdata=imgdata

    pg.display.flip()
    pg.display.update()

    for event in pg.event.get():
        if event.type==pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type==pg.MOUSEBUTTONDOWN:
            point=np.array(event.pos)/scale
            print(point)
