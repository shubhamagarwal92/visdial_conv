
# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


test_list = list(range(100))
batch_size = 10


for batch in divide_chunks(test_list, batch_size):
    print(batch)
