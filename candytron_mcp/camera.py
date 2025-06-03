import cv2
from ultralytics import YOLO

capture: cv2.VideoCapture | None = None
yolomodel: YOLO | None = None
positions = {}
positions_thresdist = 0

def init_cam(use_camera=True):
    if not use_camera:
        return True
    global capture, yolomodel
    yolomodel = YOLO('models/best-m.pt')
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        return False
    return True

def exit_cam():
    if has_camera():
        cv2.destroyAllWindows()

def has_camera():
    return yolomodel is not None

_open_window_count = 0

def acquire_scene_one(refresh:bool=False):
    global _open_window_count
    if not has_camera():
        return [('Riesen', 608.4746, 261.0399), ('Pearnut', 190.6651, 281.6597), ('Geisha', 324.3160, 287.9541), ('Dumle', 331.5356, 43.3107), ('VanillaFudge', 328.1843, 159.3306), ('Riesen', 463.4600, 432.1105), ('Refreshers', 461.0807, 162.0085), ('Riesen', 316.5636, 432.8765), ('Refreshers', 567.1455, 74.4779), ('Plopp', 197.7049, 161.3752), ('Refreshers', 466.4943, 293.8249)]
    if refresh:
        capture.read()
        capture.read()
    ret, fr = capture.read()
    res = yolomodel(fr, verbose=False)[0]

    # Visualize the results on the frame
    annotated_frame = res.plot()            
    # Display the annotated frame
    cv2.imshow('YOLO Detection', annotated_frame)
    if _open_window_count == 0:
        cv2.moveWindow('YOLO Detection', 200, 50)
    _open_window_count += 1
    
    return [(res.names[int(bx.cls)],
             (bx.xyxy[0][0]+ bx.xyxy[0][2])/2,
             (bx.xyxy[0][1]+ bx.xyxy[0][3])/2) for bx in res.boxes]

def camera_check_event():
    # Break the loop if 'q' is pressed
    return cv2.waitKey(1) & 0xFF == ord('q') if has_camera() else False


def calibrate_positions(n, m):
    # Currently assumes that letters (m) start from bottom and increase upwards
    # in the picture, and numbers (n) increase from left to right.
    global positions
    global positions_thresdist
    lst = acquire_scene_one() if has_camera() else [('Refreshers', 465.0, 43.0), ('Riesen', 465.0, 432.0), ('Plopp', 193, 43.0), ('Pearnut', 190.6651, 432.0)]
    if len(lst) != 4:
        print("Calibrate positions failed, 4 != len = " + str(len(lst)))
        return False
    lst.sort(key=lambda t: t[1])
    if lst[0][2] < lst[1][2]:
        tl = lst[0]
        bl = lst[1]
    else:
        tl = lst[1]
        bl = lst[0]
    if lst[2][2] < lst[3][2]:
        tr = lst[2]
        br = lst[3]
    else:
        tr = lst[3]
        br = lst[2]
    positions = {}
    for i in range(n):
        for j in range(m):
            tag = chr(j+65) + str(i+1)
            pos = tuple((tl[k]*(n-1-i)*j + bl[k]*(n-1-i)*(m-1-j) + tr[k]*i*j + br[k]*i*(m-1-j))/(n-1)/(m-1) for k in [1,2])
            positions[tag] = pos
    positions_thresdist = min((tr[1]+br[1]-tl[1]-bl[1])/(n-1)/4, (bl[2]+br[2]-tl[2]-tr[2])/(m-1)/4)**2
    return positions

def camera_positions():
    return positions

def find_position(xy):
    global positions
    global positions_thresdist
    mindist = positions_thresdist
    mintag = False
    for tag in positions:
        dist = (positions[tag][0]-xy[0])**2 + (positions[tag][1]-xy[1])**2
        if dist < mindist:
            mindist = dist
            mintag = tag
    return mintag

def incrdic(dic, key, val):
    if key not in dic:
        dic[key] = {}
    if val not in dic[key]:
        dic[key][val] = 1
    else:
        dic[key][val] += 1

def acquire_scene():
    dic = {}
    for i in range(5):
        lst = acquire_scene_one()
        for ele in lst:
            ptag = find_position((ele[1],ele[2]))
            if ptag:
                incrdic(dic, ptag, ele[0])
    dic2 = {}
    for ptag in sorted(dic):
        mx = 0
        mxobj = False
        for obj in dic[ptag]:
            if dic[ptag][obj] > mx:
                mx = dic[ptag][obj]
                mxobj = obj
        dic2[ptag] = mxobj
    return dic2
