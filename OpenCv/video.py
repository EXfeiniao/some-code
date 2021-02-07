import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
# eg.5
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while(True):
    # 逐帧捕获
    ret, frame = cap.read()

    # 操作框架
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 显示结果帧
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放捕获
cap.release()
cv2.destroyAllWindows()
'''

# eg.6
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('OutPut.avi', fourcc, 20.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # frame = cv2.flip(frame, 0)
        # 这个操作会使视频上下翻转

        # 写入帧
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放捕获
cap.release()
out.release()
cv2.destroyAllWindows()
