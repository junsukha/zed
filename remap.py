import cv2
import numpy as np

np.random.seed(0)
img = np.uint8(np.random.rand(8, 8)*255)
print(img)
# [[139 182 153 138 108 164 111 227]
#  [245  97 201 134 144 236  18  22]
#  [  5 212 198 221 249 203 117 199]
#  [ 30 163  36 240 133 105  67 197]
#  [116 144   4 157 156 157 240 173]
#  [ 91 111 177  15 170 171  53  32]
#  [ 80  92 145 111 252  26  53  41]
#  [166  64 118  62  40  28 167  35]]


map_y = np.array([[0, 1], [2, 3]], dtype=np.float32)
map_x = np.array([[5, 6], [7, 100]], dtype=np.float32)
mapped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

print(mapped_img)