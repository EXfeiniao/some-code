import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
# eg.7
img = np.zeros((512, 512, 3), np.uint8)

# 画线
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
# 画矩形
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
# 画圆
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)
# 画椭圆
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)
# 画多边形
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts.reshape((-1, 1, 2))
# 第一个参数为-1，表明这一维度的长度是根据后面的维度计算出来的

# 写字
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2)

# 显示
winname = 'eg.7'
cv2.namedWindow(winname)
cv2.imshow(winname, img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# eg.8
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

# eg.9
# 在图片上双击过的位置绘制一个圆圈


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_MBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('eg.9')
cv2.setMouseCallback('eg.9', draw_circle)

while(1):
    cv2.imshow('eg.9', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
'''

# eg.10
# 根据选择的模式在拖动鼠标时绘制矩形或者是圆圈(就像画图程序中一样)

# 当鼠标按下时变成True
drawing = False
# 如果mode为True绘制矩形。按下'm' 变成绘制曲线
mode = True
ix, iy = -1, -1


# 创建回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    # 当按下左键时返回起始坐标位置
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # 当按下鼠标左键移动时绘制图形，event可以查看移动，flag查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                # 绘制圆圈，小圆圈连在一起就成了线，3代表了笔画的粗细
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                # 下面注释掉的代码是起始点为圆心，起点到终点为半径的
                # r = int(np.sqrt((x - ix)**2 + (y - iy)**2))
                # cv2.circle(img, (x, y), r, (0, 0, 255), -1)
    # 当鼠标松开时停止绘画
    elif event == cv2.EVENT_LBUTTONUP:
        drawing == False
        # if mode == True:
        #     cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        # else:
        #     cv2.circle(img, (x, y), 5, (0, 0, 255), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('eg.10')
cv2.setMouseCallback('eg.10', draw_circle)
while(1):
    cv2.imshow('eg.10', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break