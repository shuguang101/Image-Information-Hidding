import random
from bitarray import bitarray


def similarity(bytes1: bytes, bytes2: bytes):
    if len(bytes1) != len(bytes2):
        return 0.0

    b1 = bitarray(endian='little')
    b2 = bitarray(endian='little')
    b1.frombytes(bytes1)
    b2.frombytes(bytes2)

    same_bit = (~(b1 ^ b2)).count()
    total_bit = len(bytes1) * 8

    return same_bit / total_bit


print(similarity(b"aa", b"ab"))
print(similarity(b"aaa", b"ab"))
