import numpy as np
import matplotlib.pyplot as plt
import cv2

# 图片路径
img_dir = './data/img/lena.png'
img = cv2.imread(img_dir, 0)

'''
# eg.1
cv2.imshow('001', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# eg.2
cv2.namedWindow('002', cv2.WINDOW_NORMAL)
cv2.imshow('002', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('test.png', img)

# eg.3
img = cv2.imread('test.png', 0)
cv2.imshow('003', img)
k = cv2.waitKey(5000) & 0xFF
if k == 27:     # 键入Esc退出
    print("way of if")
    cv2.destroyAllWindows()
elif k == ord('s'):     # 键入 's' 保存并退出
    print("way of elif")
    cv2.imwrite('test_s.png', img)
    cv2.destroyAllWindows()

# eg.4
img = cv2.imread('b1.jpg')  # 加上0就是灰度，只有一个通道

# 练习
b, g, r = cv2.split(img)
img2 = cv2.merge([r, g, b])
plt.subplot(121)
plt.imshow(img)    # bgr下错误的图片
plt.subplot(122)
plt.imshow(img2)   # 正确的颜色
plt.show()
cv2.imshow('bgr image', img)     # expects true color
cv2.imshow('rgb image', img2)    # expects distorted color
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # 隐藏xy轴上的 刻度值
# plt.show()
'''
