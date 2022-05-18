import pygame as pg
import pygame.camera as camera
from pygame.locals import *
import os,sys

import matplotlib.pyplot as plt

import numpy as np
import cv2

pg.init()
camera.init()

size=[900,900]
screen=pg.display.set_mode(size)

clock=pg.time.Clock()

cam=camera.Camera(camera.list_cameras()[0],(400,300))
cam.start()

last_imgdata=pg.surfarray.array3d(cam.get_image()).swapaxes(0,1).astype(np.float32)
last_imgdata=cv2.cvtColor(last_imgdata,cv2.COLOR_BGR2GRAY)

point=np.array([100,100],dtype=np.float32)
scale=2
    
grid=20

kernel1d=cv2.getGaussianKernel(2*grid-1,10)
kernel=np.outer(kernel1d,kernel1d.transpose())

def get_v(x,y,rt,rx,ry):

    x=round(x)
    y=round(y)

    nrt=rt[y-grid+1:y+grid,x-grid+1:x+grid]*kernel
    nrx=rx[y-grid+1:y+grid,x-grid+1:x+grid]*kernel
    nry=ry[y-grid+1:y+grid,x-grid+1:x+grid]*kernel

    a1=np.array([[np.sum(nrx*nrx),np.sum(nrx*nry)],[np.sum(nrx*nry),np.sum(nry*nry)]])
    a2=np.array([[np.sum(nrx*nrt)],[np.sum(nry*nrt)]])

    v=np.zeros((2,1))

    if np.linalg.matrix_rank(a1)==a1.shape[0]:
        v=-np.linalg.solve(a1,a2)

    return v


while True:

    dt=clock.tick(24)/1000
    screen.fill((0,0,0))
    frame=cam.get_image()
    frame_resize=pg.transform.scale(frame,scale*np.array(frame.get_size()))
    screen.blit(frame_resize,(0,0))

    imgdata=pg.surfarray.array3d(frame).swapaxes(0,1).astype(np.float32)
    imgdata=cv2.cvtColor(imgdata,cv2.COLOR_BGR2GRAY)  #240,320
    imgdata=cv2.bilateralFilter(imgdata,5,1,1)

    rt=(imgdata-last_imgdata)/dt
    rx=(np.roll(imgdata,-1,axis=1)-np.roll(imgdata,1,axis=1))/2
    ry=(np.roll(imgdata,-1,axis=0)-np.roll(imgdata,1,axis=0))/2
    
    #plt.imshow(rt,cmap='gray')
    #plt.show()
    

    for y in range(grid,imgdata.shape[0]-grid+1,grid):
        for x in range(grid,imgdata.shape[1]-grid+1,grid):

            v=get_v(x,y,rt,rx,ry)

            pg.draw.line(screen,(255,0,0),[scale*x,scale*y],[scale*x+v[0,0],scale*y+v[1,0]],2)

    #point+=get_v(point[0],point[1],rt,rx,ry)[:,0]*dt

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

