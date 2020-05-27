import h5py
from tqdm import tqdm
# https://stackoverflow.com/questions/53085947/problem-renaming-all-hdf5-datasets-in-group-for-large-hdf5-files

# Open file
with h5py.File('test.hdf5') as f:

    # Iterate over each group
    for top_key, group in f.items():
        #Rename all datasets in the group (pad with zeros)
        for key in tqdm(group.keys()):
            new_key = ("{:0<" + str(len(group)) + "}").format(str(key))
            group.move(key, new_key)
