import numpy as np

concat_list = []
a = np.random.rand(2,3)
b = np.random.rand(5,3)

concat_list.append(a)
concat_list.append(b)

concat_array = np.concatenate(concat_list)
print(concat_array.shape)

# np.array(list_of_arrays).flatten().tolist()
