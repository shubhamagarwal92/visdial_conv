import h5py
import numpy as np
path = '../data/test_small.h5'
with h5py.File(path, "w") as f:
    dset = f.create_dataset('db', (2000,10,400,768), dtype='f', chunks=(1,10,400,768))
    print(dset.shape)
    batch_size = 16
    for i in range(2000):
        if i%100 == 0:
            print(i)
        dset[i*batch_size:(i+1)*batch_size, :, :, :] = np.random.random((batch_size,10,400,768))
        # print(dset[i:(i+1),:,:])
        # print(dset.shape)
    # print(dset.chunks)

with h5py.File(path, "r") as f:
  dset = f.get('db')
  print(dset.chunks)
  print(dset.shape)
  print(dset[0,:,:,:])
  print(dset[1000,:,:,:])

