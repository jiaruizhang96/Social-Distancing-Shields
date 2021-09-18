import cv2
import numpy as np

def rectangle_detection_by_colour():
    lower_black = np.array([35, 43, 46])
    upper_black = np.array([77, 255, 255])

    img = cv2.imread('./pic/test_platform.png')

    # change to hsv model
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get mask
    mask = cv2.inRange(hsv, lower_black, upper_black)
    #cv2.imwrite('./result/Mask.png', mask)
    #mask=cv2.blur(mask,(3,3))
    mask = cv2.Canny(mask,500,1000,3, L2gradient=True)
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
    cv2.imwrite('./result/Result3.png', img)
    return rectangles