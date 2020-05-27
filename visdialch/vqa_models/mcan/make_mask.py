import torch


# Masking the sequence mask
def make_mask(feature):
    """

    :param feature:
        for img: (bs, proposals, 2048/512)
        for text - do text.unsqueeze(2) first : (bs, seq_len, 1)
    :return:
        shape: (bs, 1, 1, seq_len/proposal)
    """
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)




if __name__ == "__main__":
    # feature = torch.rand(1, 36, 2048)

    # (bs, proposal/seq_length, emb_size)
    feature = torch.rand(3,5,4)

    print(feature)
    abso = torch.abs(feature)

    print(abso)
    print(abso.size())

    sum_element = torch.sum(
        torch.abs(feature),
        dim=-1) == 0

    print(sum_element)

    mask = make_mask(feature)
    print(mask.size())
    print(mask)

    feature = torch.tensor([[[1,2,0,0],[0,4,5,6],[0,0,0,0]]])
    print(feature.size())
    abso = torch.abs(feature)
    print(abso)

    mask = make_mask(feature)
    print(mask.size())
    print(mask)
