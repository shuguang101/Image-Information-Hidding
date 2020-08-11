import cv2
import numpy as np

part1 = cv2.imread('images/part_imgs/a_part_1.png')[:, 1:-1, :]
part2 = cv2.imread('images/part_imgs/a_part_2.png')[:, 1:-1, :]
templete = cv2.imread('images/part_imgs/a_templete.png')

# part1 = cv2.imread('images/part_imgs/b_part_1.png')[:, 3:-3, :]
# part2 = cv2.imread('images/part_imgs/b_part_2.png')[:, 3:-3, :]
# templete = cv2.imread('images/part_imgs/b_templete.png')

print(part1.shape)
print(part2.shape)
print(templete.shape)

res1 = cv2.matchTemplate(part1, templete, cv2.TM_CCOEFF_NORMED)
res2 = cv2.matchTemplate(part2, templete, cv2.TM_CCOEFF_NORMED)
a1, b1 = np.where(res1 == np.max(res1))
a2, b2 = np.where(res2 == np.max(res2))
left_l_1, top_l_1, right_l_1, bottom_l_1 = b1[0], a1[0], part1.shape[1] - b1[0], part1.shape[0] - a1[0]
left_l_2, top_l_2, right_l_2, bottom_l_2 = b2[0], a2[0], part2.shape[1] - b2[0], part2.shape[0] - a2[0]

print(left_l_1, top_l_1, right_l_1, bottom_l_1)
print(left_l_2, top_l_2, right_l_2, bottom_l_2)

sx, sy = max(left_l_1, left_l_2), max(top_l_1, top_l_2)
w = max(left_l_1, left_l_2) + max(right_l_1, right_l_2)
h = max(top_l_1, top_l_2) + max(bottom_l_1, bottom_l_2)

new_img = np.zeros((h, w, 3), dtype=np.uint8)
new_img[:] = [75, 67, 60]
print(new_img.shape)
new_img[sy - a1[0]:sy - a1[0] + part1.shape[0], sx - b1[0]:sx - b1[0] + part1.shape[1], :] = part1[:]
new_img[sy - a2[0]:sy - a2[0] + part2.shape[0], sx - b2[0]:sx - b2[0] + part2.shape[1], :] = part2[:]

cv2.rectangle(part1, (b1[0], a1[0]), (b1[0] + templete.shape[1], a1[0] + templete.shape[0]), (255, 0, 0))
cv2.rectangle(part2, (b2[0], a2[0]), (b2[0] + templete.shape[1], a2[0] + templete.shape[0]), (255, 0, 0))

cv2.imshow('part1', part1)
cv2.imshow('part2', part2)
cv2.imshow('templete', templete)
cv2.imshow('new_img', new_img)

cv2.waitKey()
