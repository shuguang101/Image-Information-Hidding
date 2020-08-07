import cv2
import numpy as np
from bitarray import bitarray


# JSteg最早在JPEG图像中进行隠写的方法之一, 但该方法会使得DCT系数直方图有明显的变化 [使得DCT系数中(2i, 2i+1)的频率趋于一致]
# 容易被卡方检测出来
# 用秘密信息比特直接替换JPEG图像中量化后DCT系数的最低比特位, 对于为0或1的DCT系数不做处理
# LSB: least significant bits
# MSB: Most significant bits

# steganography 隠写


def constant_pad_8_8(image, pad_val=0):
    assert image.ndim == 2 or image.ndim == 3

    h, w = image.shape[:2]
    h_pad = 0 if h % 8 == 0 else 8 - h % 8
    w_pad = 0 if w % 8 == 0 else 8 - w % 8

    padded_image = cv2.copyMakeBorder(image, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, pad_val)
    return padded_image


def hide_the_text(image, text):
    # 将文本转化成位数组
    text_bit_array = bitarray(endian='big').frombytes(text.encode("UTF-8"))

    pass


def reveal_the_text(image):
    # text_bit_array.tobytes()
    pass


if __name__ == "__main__":
    pass
