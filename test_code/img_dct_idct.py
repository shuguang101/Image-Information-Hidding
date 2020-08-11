import cv2
import numpy as np

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


def constant_pad_8_8(image, pad_val=0):
    assert image.ndim == 2 or image.ndim == 3

    h, w = image.shape[:2]
    h_pad = 0 if h % 8 == 0 else 8 - h % 8
    w_pad = 0 if w % 8 == 0 else 8 - w % 8

    padded_image = cv2.copyMakeBorder(image, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, pad_val)
    return padded_image


def as_type_test():
    test_array = np.array([-1, 0.5, 1.3, 1, 9, 256])
    test_array_uint8 = test_array.astype(np.uint8)
    print(test_array)
    print(test_array_uint8)
    # [-1.    0.5   1.3   1.    9.  256.]
    # [255   0   1   1   9   0]


if __name__ == '__main__':
    # as_type_test()

    img_path = "images/corel_bavarian_couple.jpg"
    # img_path = "images/mark_img_210x70.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # padding
    padded_img = constant_pad_8_8(img)
    h, w = padded_img.shape[:2]

    # 反量化重构的图像
    reconstructed_img = np.zeros_like(padded_img, np.uint8)

    for hs in range(0, h, 8):
        for ws in range(0, w, 8):
            # 8x8 局部图像
            sub_img = padded_img[hs:hs + 8, ws:ws + 8]
            # dct变换 [-128, 127]
            sub_img_dct = cv2.dct(sub_img - 128.0)
            # 量化
            quantification_sub_img_dct = (sub_img_dct / light_table).astype(np.int)
            # 反量化
            i_quantification_sub_img_dct = quantification_sub_img_dct * light_table * 1.0
            # 反dct变换
            sub_img_idct = cv2.idct(i_quantification_sub_img_dct)
            # 恢复图像
            reconstructed_img[hs:hs + 8, ws:ws + 8] = np.clip(128 + sub_img_idct, 0, 255).astype(np.uint8)

            # 输出
            if hs == 0 and ws == 0:
                print("raw dct:")
                print(sub_img_dct.astype(np.int))
                print("quantification data:")
                print(quantification_sub_img_dct.astype(np.int))
                print("reconstructed dct:")
                print(i_quantification_sub_img_dct.astype(np.int))

    cv2.imshow("padded_img", padded_img)
    cv2.imshow("reconstructed_img", reconstructed_img)
    cv2.waitKey()
