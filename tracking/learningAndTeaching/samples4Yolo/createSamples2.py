

import os
import numpy as np
import pandas as pd
import cv2
import panZoom

DATAHOME = '/home/ctorney/data/tz-2017/'
CODEHOME = '/home/ctorney/workspace/wildGradient/'
inputname = CODEHOME + '/irMovieList.csv'
dfMovies = pd.read_csv(inputname,index_col=0)




# initialize the list of points for the rectangle bbox,
# the temporaray endpoint of the drawing rectangle
# the list of all bounding boxes of selected rois
# and boolean indicating wether drawing of mouse
# is performed or not
rect_endpoint_tmp = []
rect_bbox = []
bbox_list_rois = []
drawing = False

def select_rois(img):
    """
    Interactive select rectangle ROIs and store list of bboxes.
    
    Parameters
    ----------
    img :
        image 3-dim.
    
    Returns
    -------
    bbox_list_rois : list of list of int
        List of bboxes of rectangle rois.
    """
    # mouse callback function
    def draw_rect_roi(event, x, y, flags, param):
            # grab references to the global variables
            global rect_bbox, rect_endpoint_tmp, drawing
    
            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that drawing is being
            # performed. set rect_endpoint_tmp empty list.
            if event == cv2.EVENT_LBUTTONDOWN:
                rect_endpoint_tmp = []
                rect_bbox = [(x, y)]
                drawing = True
    
            # check to see if the left mouse button was released
            elif event == cv2.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # drawing operation is finished
                rect_bbox.append((x, y))
                drawing = False
    
                # draw a rectangle around the region of interest
                p_1, p_2 = rect_bbox
                cv2.rectangle(img, p_1, p_2, color=(0, 255, 0),thickness=1)
                cv2.imshow('image', img)
    
                # for bbox find upper left and bottom right points
                p_1x, p_1y = p_1
                p_2x, p_2y = p_2
    
                lx = min(p_1x, p_2x)
                ty = min(p_1y, p_2y)
                rx = max(p_1x, p_2x)
                by = max(p_1y, p_2y)
    
                # add bbox to list if both points are different
                if (lx, ty) != (rx, by):
                    bbox = [lx, ty, rx, by]
                    bbox_list_rois.append(bbox)
    
            # if mouse is drawing set tmp rectangle endpoint to (x,y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                rect_endpoint_tmp = [(x, y)]
    
    
    # clone image img and setup the mouse callback function
    img_copy = img.copy()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rect_roi)
    
    # keep looping until the 'c' key is pressed
    while True:
        # display the image and wait for a keypress
        if not drawing:
            cv2.imshow('image', img)
        elif drawing and rect_endpoint_tmp:
            rect_cpy = img.copy()
            start_point = rect_bbox[0]
            end_point_tmp = rect_endpoint_tmp[0]
            cv2.rectangle(rect_cpy, start_point, end_point_tmp,(0,255,0),1)
            cv2.imshow('image', rect_cpy)
    
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord('c'):
            break
    # close all open windows
    cv2.destroyAllWindows()
    
    return bbox_list_rois

imName = 'test.png'

frame = cv2.imread(imName)

aa= select_rois(frame)    
    



