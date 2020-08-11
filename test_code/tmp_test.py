import random
from bitarray import bitarray


random.seed(123456)
print(random.sample(range(10),5))

a= [1,2,3,4,5]
random.shuffle(a)
print(a)

bit_data_list = [1,0,0,1,1]
bit_array = bitarray(bit_data_list,endian='little')
# bit_array.append(bit_data_list)
print(bit_array)