import cv2
import random
import numpy as np
from bitarray import bitarray

#  亮度量化表
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

#  色差量化表
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


# JSteg最早在JPEG图像中进行隠写的方法之一, 但该方法会使得DCT系数直方图有明显的变化 [使得DCT系数中(2i, 2i+1)的频率趋于一致]
# 本程序采用JSteg类似的方式在DCT域进行信息隠写, 用秘密信息比特位直接替换量化后DCT系数的最低比特位(0或1的DCT系数不做处理)
# 程序使用两种方式: 8x8子图像处理方式(不支持crop), 整图处理方式
# LSB: least significant bits
# MSB: Most significant bits
# steganography 隠写

class DCTBasedLSB(object):
    def __init__(self,
                 use_sub_img: bool = True,
                 table_scale: float = 6,
                 norm_size: tuple = (512, 512),
                 seed: int = 20200811,
                 use_dc: bool = True,
                 debug: bool = False
                 ):
        assert len(norm_size) == 2

        use_light_table: bool = True

        # 是否使用8x8的子图像处理方式
        self.use_sub_img = use_sub_img
        # 量化表缩放系数
        self.table_scale = table_scale
        # 将图像缩放到标准大小进行操作
        self.norm_size = norm_size
        # 随机数种子
        self.seed = seed
        # 是否使用直流dct系数
        self.use_dc = use_dc
        # 最多支持的隠写字 位数
        self.max_bit_to_hide = ((norm_size[0] // 8) * (norm_size[1] // 8))
        # do debug
        self.debug = debug

        # 使用亮度表或色度表
        table = light_table if use_light_table else color_table
        # 根据使用的亮度表或色度表的不同选择相应的通道
        self.channel_index = 0 if use_light_table else 1

        # 获取实际使用的量化表
        if use_sub_img:
            self.used_table = np.around(table_scale * table).astype(np.int)
        else:
            self.used_table = np.around(cv2.resize(table * 1.0, norm_size) * table_scale).astype(np.int)

    # 提取秘密信息
    def reveal_the_data(self, steganography_bgr_image: np.ndarray, secret_data_len_in_bytes: int):
        # 3通道BGR图像
        assert steganography_bgr_image.ndim == 3 and steganography_bgr_image.dtype == np.uint8

        # 转换到 YCrCb(亮度 红色差 蓝色差) 颜色空间
        steganography_bgr_image_norm_size = cv2.resize(steganography_bgr_image, self.norm_size)
        steganography_yuv_image = cv2.cvtColor(steganography_bgr_image_norm_size, cv2.COLOR_BGR2YCrCb) * 1.0

        # 要提取的数据长度
        secret_data_len_in_bits = secret_data_len_in_bytes * 8
        # 提取到的信息
        bit_data_list = []

        if self.debug:
            print(id(self), "get dct_q: ")
        if self.use_sub_img:
            # 随机选择位置嵌入信息
            pos_list = []
            pos_list_index = 0
            for hs in range(0, steganography_yuv_image.shape[0] // 8 * 8, 8):
                for ws in range(0, steganography_yuv_image.shape[1] // 8 * 8, 8):
                    pos_list.append((hs, ws))
            random.seed(self.seed)
            random.shuffle(pos_list)

            # 提取秘密信息
            while len(bit_data_list) < secret_data_len_in_bits and pos_list_index < len(pos_list):
                hs, ws = pos_list[pos_list_index]
                pos_list_index += 1
                # 计算Y通道的DCT变换
                img_dct = cv2.dct(steganography_yuv_image[hs:hs + 8, ws:ws + 8, self.channel_index] - 127.5)
                # 使用亮度量化表进行量化
                img_dct_q = np.around(img_dct / self.used_table).astype(np.int)
                # 提取秘密信息
                sub_bit_data_list = self.extract_bit_data(img_dct_q, 1)
                if len(sub_bit_data_list) > 0:
                    bit_data_list.append(sub_bit_data_list[0])
        else:
            # 计算Y通道的DCT变换
            img_dct = cv2.dct(steganography_yuv_image[:, :, self.channel_index] - 127.5)
            # 使用亮度量化表进行量化
            img_dct_q = np.around(img_dct / self.used_table).astype(np.int)
            # 提取秘密信息
            bit_data_list = self.extract_bit_data(img_dct_q, secret_data_len_in_bits)

        if self.debug:
            print("bit len: ", bit_data_list, secret_data_len_in_bits)
        assert len(bit_data_list) == secret_data_len_in_bits

        if self.debug:
            print()

        # 对秘密信息字节进行转换
        bit_array = bitarray(bit_data_list, endian='little')
        # 返回秘密信息
        return bit_array.tobytes()

    # 嵌入秘密信息
    def steganography_the_data(self, bgr_image: np.ndarray, secret_data: bytes):
        # 3通道BGR图像
        assert bgr_image.ndim == 3 and bgr_image.dtype == np.uint8
        # 最多能存储 max_bit_to_hide bit数据
        assert len(secret_data) * 8 < self.max_bit_to_hide

        # 图片原始大小
        raw_h, raw_w = bgr_image.shape[:2]

        # 转换到 YCrCb(亮度 红色差 蓝色差) 颜色空间, dtype=uint8
        yuv_image = cv2.cvtColor(cv2.resize(bgr_image, self.norm_size), cv2.COLOR_BGR2YCrCb) * 1.0

        # 嵌入秘密信息的图像
        steganography_yuv_image = np.zeros_like(yuv_image, np.float64)
        steganography_yuv_image[:] = yuv_image[:]

        # 对秘密信息字节进行转换
        bit_array = bitarray(endian='little')
        bit_array.frombytes(secret_data)
        bit_data_list = bit_array.tolist()
        bit_data_insert_count = 0

        if self.debug:
            print(id(self), "add dct_q: ")

        if self.use_sub_img:
            # 随机选择位置嵌入信息
            pos_list = []
            pos_list_index = 0
            for hs in range(0, yuv_image.shape[0] // 8 * 8, 8):
                for ws in range(0, yuv_image.shape[1] // 8 * 8, 8):
                    pos_list.append((hs, ws))
            random.seed(self.seed)
            random.shuffle(pos_list)

            # 嵌入秘密信息
            while bit_data_insert_count < len(bit_data_list) and pos_list_index < len(pos_list):
                hs, ws = pos_list[pos_list_index]
                pos_list_index += 1
                # 计算Y通道的DCT变换
                img_dct = cv2.dct(steganography_yuv_image[hs:hs + 8, ws:ws + 8, self.channel_index] - 127.5)
                # 使用亮度量化表进行量化
                img_dct_q = np.around(img_dct / self.used_table).astype(np.int)
                # 尝试嵌入秘密信息位
                bit_data_insert_count += self.insert_bit_data(img_dct_q, [bit_data_list[bit_data_insert_count]])
                # 反量化, 反dct变换, 恢复图像
                steganography_yuv_image[hs:hs + 8, ws:ws + 8, 0] = 127.5 + cv2.idct(img_dct_q * self.used_table * 1.0)
        else:
            # 计算Y通道的DCT变换
            img_dct = cv2.dct(steganography_yuv_image[:, :, self.channel_index] - 127.5)
            # 使用亮度量化表进行量化
            img_dct_q = np.around(img_dct / self.used_table).astype(np.int)
            # 尝试嵌入秘密信息位
            bit_data_insert_count += self.insert_bit_data(img_dct_q, bit_data_list)
            # 反量化, 反dct变换, 恢复图像
            steganography_yuv_image[:, :, self.channel_index] = 127.5 + cv2.idct(img_dct_q * self.used_table * 1.0)

        if self.debug:
            print("bit len: ", len(bit_data_list), bit_data_insert_count)
        # 校验嵌入的数据量
        assert bit_data_insert_count == len(bit_data_list)

        # 转化回BGR图像
        steganography_yuv_image = np.clip(steganography_yuv_image, 0, 255).astype(np.uint8)
        steganography_bgr_image = cv2.cvtColor(steganography_yuv_image, cv2.COLOR_YCrCb2BGR)
        steganography_bgr_image = cv2.resize(steganography_bgr_image, (raw_w, raw_h))

        if self.debug:
            print()

        return steganography_bgr_image

    @staticmethod
    def find_significant_components(dct_data, do_sort=False):
        if do_sort:
            dct_data_abs = np.abs(dct_data)
            index_array = np.argsort(-dct_data_abs, None)  # default asc
            index_tuple = np.unravel_index(index_array, dct_data_abs.shape)
        else:
            i_list, j_list = [], []
            for i in range(0, dct_data.shape[0], 1):
                for j in range(0, dct_data.shape[1], 1):
                    i_list.append(i)
                    j_list.append(j)
            index_tuple = (i_list, j_list)
        return index_tuple

    # 嵌入'位数据'
    def insert_bit_data(self, dct_q: np.ndarray, bit_data: list):
        # 成功嵌入的位数
        inserted_bit_count = 0

        # 排序寻找最显著的成分
        index_tuple = self.find_significant_components(dct_q)
        # 将秘密信息位嵌入 '绝对值非0,1的系数' 中
        for index in range(len(index_tuple[0])):
            i, j = index_tuple[0][index], index_tuple[1][index]
            is_dc = i == 0 and j == 0
            if dct_q[i, j] != 0 and abs(dct_q[i, j]) != 1 and (self.use_dc or not is_dc):
                sign = -1 if dct_q[i, j] < 0 else 1
                dct_q[i, j] = ((abs(dct_q[i, j]) & 0xfffffffe) | bit_data[inserted_bit_count]) * sign
                if self.debug:
                    print("%3d" % dct_q[i, j], end=" ")
                inserted_bit_count += 1
                if inserted_bit_count >= len(bit_data):
                    return inserted_bit_count

        return inserted_bit_count

    # 提取'位数据'
    def extract_bit_data(self, dct_q, bit_len):
        # 成功提取到的位数据
        extracted_bit_list = []

        # 排序寻找最显著的成分
        index_tuple = self.find_significant_components(dct_q)
        for index in range(len(index_tuple[0])):
            i, j = index_tuple[0][index], index_tuple[1][index]
            is_dc = i == 0 and j == 0
            if dct_q[i, j] != 0 and abs(dct_q[i, j]) != 1 and (self.use_dc or not is_dc):
                if self.debug:
                    print("%3d" % dct_q[i, j], end=" ")
                extracted_bit_list.append(abs(dct_q[i, j]) & 0x00000001)
                if len(extracted_bit_list) >= bit_len:
                    return extracted_bit_list
        return extracted_bit_list

    @staticmethod
    def similarity(bytes1: bytes, bytes2: bytes):
        # 计算相似度: 相同的位 / 总的位数
        if len(bytes1) != len(bytes2):
            return 0.0

        b1 = bitarray(endian='little')
        b2 = bitarray(endian='little')
        b1.frombytes(bytes1)
        b2.frombytes(bytes2)

        same_bit = (~(b1 ^ b2)).count()
        total_bit = len(bytes1) * 8

        return same_bit / total_bit
