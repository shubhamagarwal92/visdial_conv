# https://discuss.pytorch.org/t/how-to-do-padding-based-on-lengths/24442/9
# https://stackoverflow.com/a/7370980/3776827
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


# Masking the sequence mask
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


def get_nonzero():
    l = [torch.tensor([1,2,3]), torch.tensor([4,5]),torch.tensor([6,7,8,9])]
    emb_len=4
    # this is what you want:
    lp = torch.stack([torch.cat([i, i.new_zeros(emb_len - i.size(0))], 0) for i in l],1)


    # for i, tensor in enumerate(sequences):
    #     length = tensor.size(0)
    #     # use index notation to prevent duplicate references to the tensor
    #     out_tensor[i, :length, ...] = tensor


def second():
    FE_DIM = 100
    first_embed = torch.randn(5,20,FE_DIM)
    second_embed = torch.randn(5, 10)
    original_len = [10, 8, 9, 7, 3]
    PADDING_LEN = 20

    res2 = torch.zeros(5, 20, FE_DIM+10)
    res2[:,:,:FE_DIM] = first_embed
    for i, e in enumerate(second_embed):
        res2[i, :original_len[i], FE_DIM:] = e
    return res2


def test_raw_concat_hist_q():
    bs = 3
    h_len = 10
    q_len = 5
    total_len = h_len + q_len
    hist_len = [7,5,3]
    ques_len = [3,4,3]

    hist = torch.randn(bs,h_len)
    ques = torch.randn(bs,q_len)

    device = ques.device
    combined_raw_h_q = torch.zeros(bs, total_len).to(device)
    for i in range(len(ques)):
        combined_raw_h_q[i,:hist_len[i]] = hist[i][:hist_len[i]]
        print(combined_raw_h_q)
        print(ques[i][:ques_len[i]])
        combined_raw_h_q[i,hist_len[i]:hist_len[i]+ques_len[i]] = ques[i][:ques_len[i]]

    print(hist)
    print(ques)
    print(combined_raw_h_q)



if __name__ == "__main__":

    test_raw_concat_hist_q()



    # feature = torch.rand(1, 36, 2048)
    # feature = torch.tensor([[1,2,0,0],[8,4,5,6],[1,0,0,0]])
    #
    # non = feature[feature.nonzero()]
    # print("Non", non)
    #
    # length = torch.tensor([2,4,1])
    #
    # print(feature)
    # abso = torch.abs(feature)
    # print(abso)
    # y = (feature == 0).nonzero()
    # print(y)
    #
    # lengths = (torch.sum(feature, dim = -1) != 0)
    # print(lengths)
    #
    # print(feature.size())
    # print(length.size())
    #
    # pack = pack_padded_sequence(feature, length, batch_first=True, enforce_sorted=False)
    #
    # print(pack)
    #
    # print(torch.nonzero(feature))
    # y = (feature != 0)
    # print(y)
    # print(y.size())
    # y = y.nonzero()
    # print(y)
    # print(y.size())

#     feature = torch.rand(1,3,4)
#
#
#     mask = make_mask(feature)
#     print(mask.size())
#     print(mask)
