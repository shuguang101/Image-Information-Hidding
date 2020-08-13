import cv2
from bitarray import bitarray
from image_steganography.dct_based_lsb import DCTBasedLSB

if __name__ == "__main__":
    # 秘密信息
    secret_text = "you can't seen this"
    secret_data = secret_text.encode("UTF-8", 'ignore')

    # 原始图片
    raw_img = cv2.imread("images/corel_bavarian_couple.jpg")
    # raw_img = cv2.imread("images/win98.jpg")

    # 构造lsb模型
    dct_lsb_1 = DCTBasedLSB(use_sub_img=True, use_dc=True, debug=True)
    dct_lsb_2 = DCTBasedLSB(use_sub_img=False, use_dc=True, debug=True)

    # 嵌入秘密信息
    steganography_bgr_image_1 = dct_lsb_1.steganography_the_data(raw_img, secret_data)
    steganography_bgr_image_2 = dct_lsb_2.steganography_the_data(raw_img, secret_data)

    # 写入文件
    cv2.imwrite("images/steganography_bgr_image_1.jpg", steganography_bgr_image_1, [cv2.IMWRITE_JPEG_QUALITY, 15])
    cv2.imwrite("images/steganography_bgr_image_2.jpg", steganography_bgr_image_2, [cv2.IMWRITE_JPEG_QUALITY, 15])

    # 读取秘密信息图片
    steganography_bgr_image_1 = cv2.imread("images/steganography_bgr_image_1.jpg")[:, :, :3]
    steganography_bgr_image_2 = cv2.imread("images/steganography_bgr_image_2.jpg")[:, :, :3]

    # 获取秘密信息
    secret_data_extracted_1 = dct_lsb_1.reveal_the_data(steganography_bgr_image_1, len(secret_data))
    secret_data_extracted_2 = dct_lsb_2.reveal_the_data(steganography_bgr_image_2, len(secret_data))

    # 解码为字符串
    secret_text_extracted_1 = secret_data_extracted_1.decode("UTF-8", 'ignore')
    secret_text_extracted_2 = secret_data_extracted_2.decode("UTF-8", 'ignore')

    print("len=%d" % (len(secret_data) * 8))
    print(dct_lsb_1.similarity(secret_data, secret_data_extracted_1), secret_text_extracted_1)
    print(dct_lsb_2.similarity(secret_data, secret_data_extracted_2), secret_text_extracted_2)
