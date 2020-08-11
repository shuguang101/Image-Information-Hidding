import cv2
import numpy as np
from bitarray import bitarray

# JSteg最早在JPEG图像中进行隠写的方法之一, 但该方法会使得DCT系数直方图有明显的变化 [使得DCT系数中(2i, 2i+1)的频率趋于一致]
# 容易被卡方检测出来
# 用秘密信息比特直接替换JPEG图像中量化后DCT系数的最低比特位, 对于为0或1的DCT系数不做处理
# LSB: least significant bits
# MSB: Most significant bits

# steganography 隠写

#  亮度亮度量化表
light_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

#  色度量化表
color_table = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
])


def insert_one_bit(dct_q, bit_data, n=2, use_dc=True):
    # 将秘密信息位嵌入 '绝对值非0,1的系数' 中
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            is_dc = i == 0 and j == 0
            if dct_q[i, j] != 0 and abs(dct_q[i, j]) != 1 and (use_dc or not is_dc):
                sign = -1 if dct_q[i, j] < 0 else 1
                dct_q[i, j] = ((abs(dct_q[i, j]) & 0xfffffffe) | bit_data) * sign
                return True
    return False


def extract_one_bit(dct_q, n=2, use_dc=True):
    # 提取秘密信息位
    for i in range(0, n, 1):
        for j in range(0, n, 1):
            is_dc = i == 0 and j == 0
            if dct_q[i, j] != 0 and abs(dct_q[i, j]) != 1 and (use_dc or not is_dc):
                return True, abs(dct_q[i, j]) & 0x00000001
    return False, -1


def hide_the_text(image, secret_data, mul=3):
    # 调整量化表
    inner_light_table = np.around(light_table * mul)

    # 3通道 float32图像
    assert image.ndim == 3 and (image.dtype == np.float32 or image.dtype == np.float64)
    assert isinstance(secret_data, bytes)

    # 将文本转化成位数组
    text_bit_array = bitarray(endian='little')
    text_bit_array.frombytes(secret_data)
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
            img_dct_bq = np.around(img_dct_b / inner_light_table).astype(np.int)
            img_dct_gq = np.around(img_dct_g / inner_light_table).astype(np.int)
            img_dct_rq = np.around(img_dct_r / inner_light_table).astype(np.int)

            # 将秘密信息位嵌入 '绝对值非0,1的系数' 中
            if insert_bit_data_index < text_bit_array.shape[0]:
                insert_bit_data_index += insert_one_bit(img_dct_bq, text_bit_array[insert_bit_data_index])
            if insert_bit_data_index < text_bit_array.shape[0]:
                insert_bit_data_index += insert_one_bit(img_dct_gq, text_bit_array[insert_bit_data_index])
            if insert_bit_data_index < text_bit_array.shape[0]:
                insert_bit_data_index += insert_one_bit(img_dct_rq, text_bit_array[insert_bit_data_index])

            # 反量化, 反dct变换, 恢复图像
            steganography_image[hs:hs + 8, ws:ws + 8, 0] = 127.5 + cv2.idct(img_dct_bq * inner_light_table * 1.0)
            steganography_image[hs:hs + 8, ws:ws + 8, 1] = 127.5 + cv2.idct(img_dct_gq * inner_light_table * 1.0)
            steganography_image[hs:hs + 8, ws:ws + 8, 2] = 127.5 + cv2.idct(img_dct_rq * inner_light_table * 1.0)

    return steganography_image


def reveal_the_text(steganography_image, data_len_in_bit, mul=3):
    # 调整量化表
    inner_light_table = np.around(light_table * mul)

    # 3通道 float32图像
    assert steganography_image.ndim == 3
    assert steganography_image.dtype == np.float32 or steganography_image.dtype == np.float64
    assert isinstance(data_len_in_bit, int)

    bit_data_array = bitarray(endian='little')  # big little
    extract_bit_data_index = 0

    # 嵌入秘密信息
    for hs in range(0, steganography_image.shape[0] // 8 * 8, 8 * 3):
        for ws in range(0, steganography_image.shape[1] // 8 * 8, 8 * 3):
            # 对 8x8 局部图像进行 dct 变换 [-127.5, 127.5]
            img_dct_b = cv2.dct(steganography_image[hs:hs + 8, ws:ws + 8, 0] - 127.5)
            img_dct_g = cv2.dct(steganography_image[hs:hs + 8, ws:ws + 8, 1] - 127.5)
            img_dct_r = cv2.dct(steganography_image[hs:hs + 8, ws:ws + 8, 2] - 127.5)
            # 使用亮度亮度量化表进行量化
            img_dct_bq = np.around(img_dct_b / inner_light_table).astype(np.int)
            img_dct_gq = np.around(img_dct_g / inner_light_table).astype(np.int)
            img_dct_rq = np.around(img_dct_r / inner_light_table).astype(np.int)
            # 提取秘密信息
            if extract_bit_data_index < data_len_in_bit:
                is_extracted, bit_data = extract_one_bit(img_dct_bq)
                extract_bit_data_index += is_extracted
                if is_extracted:
                    bit_data_array.append(bit_data)
            if extract_bit_data_index < data_len_in_bit:
                is_extracted, bit_data = extract_one_bit(img_dct_gq)
                extract_bit_data_index += is_extracted
                if is_extracted:
                    bit_data_array.append(bit_data)
            if extract_bit_data_index < data_len_in_bit:
                is_extracted, bit_data = extract_one_bit(img_dct_rq)
                extract_bit_data_index += is_extracted
                if is_extracted:
                    bit_data_array.append(bit_data)

    return bit_data_array.tobytes()


if __name__ == "__main__":
    # 读取图像
    raw_img = cv2.imread("images/corel_bavarian_couple.jpg")[:, :, :3]

    # 秘密信息
    raw_data = "you can't seen this text".encode("UTF-8")

    # 嵌入字符串
    steganography_img = hide_the_text(raw_img * 1.0, raw_data, 17)
    # 0-100
    cv2.imwrite("steganography_img.jpg", steganography_img, [cv2.IMWRITE_JPEG_QUALITY, 20])

    # 读取隠写图像
    steganography_img = cv2.imread("steganography_img.jpg")[:, :, :3]
    # 提取秘密信息
    extracted_data = reveal_the_text(steganography_img * 1.0, len(raw_data) * 8, 17)

    # 打印提取到的信息
    print(raw_data == extracted_data)
    print("r: ", raw_data.decode("UTF-8"))
    print("e: ", extracted_data.decode("UTF-8"))

    # 显示图像
    cv2.imshow("raw_img", raw_img)
    cv2.imshow("steganography_img", steganography_img)
    cv2.waitKey()
