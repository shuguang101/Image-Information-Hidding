import cv2
import numpy as np


def ensure_greater_than_given_shape(img_rgb, mat_shape):
    height = mat_shape[0]
    width = mat_shape[1]
    # 水印图像规定大小  height x width
    if img_rgb.shape[0] < height:  # 高度小于height, 需要调整图片大小
        new_width = int(img_rgb.shape[1] * (height / img_rgb.shape[0]))
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        img_rgb = cv2.resize(img_rgb, (new_width, height))
    if img_rgb.shape[1] < width:  # 宽度小于width, 需要调整图片大小
        new_height = int(img_rgb.shape[0] * (width / img_rgb.shape[1]))
        new_height = new_height if new_height % 2 == 0 else new_height + 1
        img_rgb = cv2.resize(img_rgb, (width, new_height))
    return img_rgb


def find_significant_components(dct_data, num_to_take):
    dct_data_abs = np.abs(dct_data)
    # 排除直流成分
    dct_data_abs[0, 0] = 0

    index_array = np.argsort(-dct_data_abs, None)  # default asc
    index_tuple = np.unravel_index(index_array, dct_data_abs.shape)
    index_tuple = (index_tuple[0][:num_to_take], index_tuple[1][:num_to_take])

    return index_tuple


def water_mark_insert(raw_img_bgr, mark_data, alpha=0.1):
    assert raw_img_bgr.ndim == 3
    assert mark_data.ndim == 1

    # 最终添加了水印的3通道图像
    final_marked_img = np.zeros_like(raw_img_bgr, np.uint8)
    # 依次处理各个通道
    for channel_index in range(3):
        # 对原始图像进行DCT变换
        raw_img_rgb_dct = cv2.dct(np.float32(raw_img_bgr[:, :, channel_index]))
        # 寻找 perceptually significant components
        ind = find_significant_components(raw_img_rgb_dct, mark_data.shape[0])
        # 添加水印信息
        raw_img_rgb_dct[ind] *= (1 + mark_data * alpha)
        # 恢复图像
        final_marked_img[:, :, channel_index] = cv2.idct(raw_img_rgb_dct).astype(np.uint8)

    # 返回添加水印后的图像
    return final_marked_img


def water_mark_extract(raw_img_bgr, marked_img_bgr, mark_data_length, alpha=0.1):
    assert raw_img_bgr.ndim == 3
    assert marked_img_bgr.ndim == 3
    assert raw_img_bgr.shape == marked_img_bgr.shape

    # 水印数据
    final_mark_img = np.zeros((mark_data_length, 3), np.float32)
    # 依次在各个通道进行提取
    for channel_index in range(3):
        # 对原始图像进行DCT变换
        raw_img_rgb_dct = cv2.dct(np.float32(raw_img_bgr[:, :, channel_index]))
        # 对含有水印的图像进行DCT变换
        marked_img_rgb_dct = cv2.dct(np.float32(marked_img_bgr[:, :, channel_index]))

        # 寻找 perceptually significant components
        ind = find_significant_components(raw_img_rgb_dct, mark_data_length)

        extracted_data = ((marked_img_rgb_dct[ind] / raw_img_rgb_dct[ind]) - 1) / alpha

        final_mark_img[:, channel_index] = extracted_data

    return final_mark_img


if __name__ == "__main__":
    raw_img = cv2.imread('water_mark/corel_bavarian_couple.jpg')[:, :, :3]

    mark_data_length = 15
    mark_data = np.random.randn(mark_data_length)
    marked_img = water_mark_insert(raw_img, mark_data, 0.1)
    cv2.imshow("marked_img", marked_img)

    mark_img_extracted = water_mark_extract(raw_img, marked_img, mark_data_length, 0.1)

    print(np.linalg.norm(mark_img_extracted[:, 0] - mark_data))
    print(np.linalg.norm(mark_img_extracted[:, 1] - mark_data))
    print(np.linalg.norm(mark_img_extracted[:, 2] - mark_data))

    cv2.waitKey()
