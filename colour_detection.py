import cv2
import numpy as np
import operator
def rectangle_detection_by_colour():
    # lower_black = np.array([35, 43, 46])
    # upper_black = np.array([77, 255, 255])
    lower_black = np.array([30, 0, 0])
    upper_black= np.array([40, 100, 100])

    img = cv2.imread('htn_parkway_pic.jpg')

    # change to hsv model
    # hsv = img
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get mask
    mask = cv2.inRange(hsv, lower_black, upper_black)
    #cv2.imwrite('./result/Mask.png', mask)
    #mask=cv2.blur(mask,(3,3))
    mask = cv2.Canny(mask,500,1000,3, L2gradient=True)
    cv2.imshow("mask",mask)
    cv2.waitKey(0)
    cv2.imwrite('./result/Mask3.png', mask)
    # detect red
    res = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imwrite('./result/Result.png', res)
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.dilate(binary,None,iterations=2)
    contours, hiera = cv2.findContours(binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if len(approx) == 4:
            x, y , w, h = cv2.boundingRect(approx)
            aspectRatio = float(w)/h
            print(aspectRatio)
            if aspectRatio < 0.6 or aspectRatio > 1.4:
                cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
                cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
                rectangles.append([cX, cY])
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.imwrite('./result/Result3.png', img)
    return rectangles

# rectangle_detection_by_colour()

def rectangle_detection_by_line_drawing():
    img = cv2.imread('htn_parkway_pic.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('img',gray)
    cv2.waitKey(0)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    cv2.imshow('line_image',line_image)
    cv2.waitKey(0)

# rectangle_detection_by_line_drawing()

def rectangle_detection_by_manual_labelling():
    global initialPoint,img, preview,line_pxl_coord_arr,each_line_pxl_arr      
    
    # create black canvas of size 600x600
    # img =  np.zeros((600, 600, 3), dtype=np.uint8)
    img = cv2.imread('htn_parkway_pic.jpg')
    # intialize values in unusable states
    preview = None
    initialPoint = (-1, -1)
    line_pxl_coord_arr = []
    each_line_pxl_arr = []  
    
    # set the named window and callback          
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", drawLine)

    while (True):
        # if we are drawing show preview, otherwise the image
        if preview is None:
            cv2.imshow('image',img)
        else :
            cv2.imshow('image',preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break;

    cv2.destroyAllWindows()
    print("line_pxl_coord_arr",len(line_pxl_coord_arr),len(line_pxl_coord_arr[0]),line_pxl_coord_arr)
    
    for line_coord in line_pxl_coord_arr:
        print("line_coord",line_coord)
        mid_x = (1/2)*(min(line_coord)[0]+max(line_coord)[0])
        mid_y = (1/2)*(min(line_coord)[1]+max(line_coord)[1])
        # print(min(line_coord[0]),max(line_coord[0]))
        # print(min(line_coord[1]),max(line_coord[1]))
        circle_radius = int(max(line_coord)[0] - mid_x)
        print((int(mid_y),int(mid_x)))
        img = cv2.circle(img, (int(mid_x),int(mid_y)), circle_radius, (255,0,0), 3)
        # Displaying the image
        cv2.imshow("img", img)
        cv2.waitKey(0)


# mouse callback
def drawLine(event,x,y,flags,param):
    global initialPoint,img, preview,line_pxl_coord_arr,each_line_pxl_arr      
        
    if event == cv2.EVENT_LBUTTONDOWN:
        # new initial point and preview is now a copy of the original image
        initialPoint = (x,y)
        preview = img.copy()
        # this will be a point at this point in time
        cv2.line(preview, initialPoint, (x,y), (0,255,0), 3)
        # print("(x,y)",(x,y),each_line_pxl_arr)
        each_line_pxl_arr = [] 
        each_line_pxl_arr.append((x,y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if preview is not None:
            # copy the original image again a redraw the new line
            preview = img.copy()
            cv2.line(preview, initialPoint, (x,y), (0,255,0), 3)
            # print("(x,y)2",(x,y),each_line_pxl_arr)
            each_line_pxl_arr.append((x,y))

    elif event == cv2.EVENT_LBUTTONUP:
        # if we are drawing, preview is not None and since we finish, draw the final line in the image
        if preview is not None:
                preview = None
                cv2.line(img, initialPoint, (x,y), (255,0,0), 3)
                # print("(x,y)3",(x,y),each_line_pxl_arr,line_pxl_coord_arr)
                each_line_pxl_arr.append((x,y))
                line_pxl_coord_arr.append(each_line_pxl_arr)
rectangle_detection_by_manual_labelling()
