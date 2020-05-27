import h5py


def save_h5_data(file_path: str, data_split: str, **data_dump):

    hf = h5py.File(file_path, 'w')
    hf.attrs['split'] = data_split

    for key in data_dump:
        hf.create_dataset(key, data=data_dump[key])

    hf.close()
