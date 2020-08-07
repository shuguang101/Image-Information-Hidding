import cv2


# JSteg最早在JPEG图像中进行隠写的方法之一, 但该方法会使得DCT系数直方图有明显的变化 [使得DCT系数中(2i, 2i+1)的频率趋于一致]
# 容易被卡方检测出来
# 用秘密信息比特直接替换JPEG图像中量化后DCT系数的最低比特位, 对于为0或1的DCT系数不做处理
# LSB: least significant bits
# MSB: Most significant bits

# steganography 隠写


def hide_the_text(image, text):
    pass


def reveal_the_text(image):
    pass


if __name__ == "__main__":
    pass
