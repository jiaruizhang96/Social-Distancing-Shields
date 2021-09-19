#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
from requests import request
#Initialize the Flask app
app = Flask(__name__)
# test_video = r'C:\Users\kathy\Documents\SD-shields-hackthenorth\Social-Distancing-Shields\filename_v1.avi'
test_video = r'C:\Users\kathy\Documents\SD-shields-hackthenorth\Social-Distancing-Shields\parkway_1.mp4'

camera = cv2.VideoCapture(test_video)
'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''
def gen_frames():
    cnt = 0  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if cnt == 0:
                cv2.imwrite('captured_img.jpg',frame)
            cnt +=1
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
@app.route('/')
def index():
    # return render_template('index.html')
    return render_template('homepage.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/vmd_timestamp')
def vmd_timestamp():
    return render_template('videopage.html')

@app.route('/capture')
def capture():
    img = cv2.imread('captured_img.jpg')
    rectangle_detection_by_manual_labelling(img)
    return render_template('capture.html')


def rectangle_detection_by_manual_labelling(input_img):
    global initialPoint,img, preview,line_pxl_coord_arr,each_line_pxl_arr      
    
    # create black canvas of size 600x600
    # img =  np.zeros((600, 600, 3), dtype=np.uint8)
    # img = cv2.imread('htn_parkway_pic.jpg')
    img = input_img
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

        ################ can be changed ########################
        safety_distance_half = int(2/3*circle_radius)
        ################ can be changed ########################

        # img = cv2.circle(img, (int(mid_x),int(mid_y)), int(circle_radius), (255,0,0), 3)
        img = cv2.rectangle(img,(min(line_coord)[0],min(line_coord)[1]-safety_distance_half),(max(line_coord)[0],max(line_coord)[1]+safety_distance_half),(0,0,255), 3)
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

if __name__ == "__main__":
    app.run(debug=True)