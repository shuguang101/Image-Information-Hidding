import cv2
import numpy as np
from bitarray import bitarray
from jpeg_table import light_table


# JSteg最早在JPEG图像中进行隠写的方法之一, 但该方法会使得DCT系数直方图有明显的变化 [使得DCT系数中(2i, 2i+1)的频率趋于一致]
# 容易被卡方检测出来
# 用秘密信息比特直接替换JPEG图像中量化后DCT系数的最低比特位, 对于为0或1的DCT系数不做处理
# LSB: least significant bits
# MSB: Most significant bits

# steganography 隠写


def insert_one_bit(sub_img, bit_data):
    is_inserted = False
    # 将秘密信息位嵌入 '绝对值非0,1的系数' 中
    for i in range(0, 8, 1):
        for j in range(0, 8, 1):
            if not is_inserted and sub_img[i, j] != 0 and abs(sub_img[i, j]) != 1:
                sign = -1 if sub_img[i, j] < 0 else 1
                print('---------------', bit_data, sub_img.dtype)
                print(sub_img[i, j])
                sub_img[i, j] = ((abs(sub_img[i, j]) & 0xfffffffe) | bit_data) * sign
                print(sub_img[i, j])
                is_inserted = True
    return is_inserted


def hide_the_text(image, text):
    # 3通道 float32图像
    assert image.ndim == 3 and (image.dtype == np.float32 or image.dtype == np.float64)

    # 将文本转化成位数组
    text_bit_array = bitarray(endian='big')
    text_bit_array.frombytes(text.encode("UTF-8"))
    text_bit_array = np.array(text_bit_array.tolist()).astype(np.int)
    insert_bit_data_index = 0

    # 嵌入秘密信息的图像
    steganography_image = np.zeros_like(image, np.float32)
    steganography_image[:] = image[:]

    # 嵌入秘密信息
    for hs in range(0, image.shape[0] // 8 * 8, 8 * 3):
        for ws in range(0, image.shape[1] // 8 * 8, 8 * 3):
            # 对 8x8 局部图像进行 dct 变换 [-127.5, 127.5]
            img_dct_b = cv2.dct(image[hs:hs + 8, ws:ws + 8, 0] - 127.5)
            img_dct_g = cv2.dct(image[hs:hs + 8, ws:ws + 8, 1] - 127.5)
            img_dct_r = cv2.dct(image[hs:hs + 8, ws:ws + 8, 2] - 127.5)
            # 使用亮度亮度量化表进行量化
            img_dct_bq = np.around(img_dct_b / light_table).astype(np.int)
            img_dct_gq = np.around(img_dct_g / light_table).astype(np.int)
            img_dct_rq = np.around(img_dct_r / light_table).astype(np.int)

            # 将秘密信息位嵌入 '绝对值非0,1的系数' 中
            if insert_bit_data_index < text_bit_array.shape[0]:
                insert_bit_data_index += insert_one_bit(img_dct_bq, text_bit_array[insert_bit_data_index])
            if insert_bit_data_index < text_bit_array.shape[0]:
                insert_bit_data_index += insert_one_bit(img_dct_gq, text_bit_array[insert_bit_data_index])
            if insert_bit_data_index < text_bit_array.shape[0]:
                insert_bit_data_index += insert_one_bit(img_dct_rq, text_bit_array[insert_bit_data_index])

            # 反量化, 反dct变换, 恢复图像
            steganography_image[hs:hs + 8, ws:ws + 8, 0] = 127.5 + cv2.idct(img_dct_bq * light_table * 1.0)
            steganography_image[hs:hs + 8, ws:ws + 8, 1] = 127.5 + cv2.idct(img_dct_gq * light_table * 1.0)
            steganography_image[hs:hs + 8, ws:ws + 8, 2] = 127.5 + cv2.idct(img_dct_rq * light_table * 1.0)

    return steganography_image


# def reveal_the_text(steganography_image, length=192):
#     assert steganography_image.ndim == 2 or steganography_image.ndim == 3
#     if steganography_image.ndim == 2:
#         steganography_image = np.expand_dims(steganography_image, -1)
#
#     h, w, c = steganography_image.shape[:3]
#     text_bit_array = bitarray(endian='big')
#     index = 0
#
#     for cs in range(0, c, 1):
#         for hs in range(0, h // 8 * 8, 8):
#             for ws in range(0, w // 8 * 8, 8):
#                 # 对 8x8 局部图像进行 dct 变换 [-128, 127]
#                 img_dct = cv2.dct(steganography_image[hs:hs + 8, ws:ws + 8, cs] - 128.0)
#                 # 使用亮度亮度量化表进行量化
#                 img_dct_q = (img_dct / light_table).astype(np.int)
#
#                 # 将秘密信息位嵌入 '绝对值非0,1的系数' 中
#                 for i in range(0, 8, 1):
#                     for j in range(0, 8, 1):
#                         if index < length \
#                                 and (i != 0 and j != 0) \
#                                 and img_dct_q[i, j] != 0 \
#                                 and abs(img_dct_q[i, j]) != 1:
#                             text_bit_array.append(img_dct_q[i, j] & 0x01)
#                             index += 1
#                             i = 8
#                             j = 8
#                             if index == 1:
#                                 print(cs, hs, ws, i, j)
#                                 print(img_dct_q)
#
#     print(np.array(text_bit_array.tolist()).astype(np.int))
#     return text_bit_array.tobytes().decode("UTF-8")


if __name__ == "__main__":
    # 读取图像
    img = cv2.imread("images/corel_bavarian_couple.jpg")[:, :, :3]

    # print((img * 1.0).dtype)
    # 嵌入字符串
    steganography_img = hide_the_text(img * 1.0, "you can't seen this text")
    cv2.imwrite("steganography_img.jpg", steganography_img)
    # #
    # # print(reveal_the_text(steganography_img, 192))
    #
    # cv2.waitKey()
