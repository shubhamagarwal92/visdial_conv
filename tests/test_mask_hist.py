import torch



def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.
    We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).long()


def masking_try(num_rounds=10, batch_size=8):
    indexes = torch.arange(1, num_rounds + 1)
    print(indexes)

    num_rounds = 10

    hist_img_mcan_embedding = torch.randn(2*10,10,5)
    batch_size, num_rounds, emb_size = hist_img_mcan_embedding.size()


    # h (bs, round, emb_size) - (0, 1, 512),  (0 ,1 512) - (0, 2, 512)
    # h (bs*round , round, emb_size)
    # q (bs, round, emb_size) - (0, 1, 512), (0,2,512) ....




    # print(hist_img_mcan_embedding)

    # mask_ = (num_rounds, num_rounds, emb_size)
    # mask_ = mask_.repeat(batch_size, 1, 1)

    # hist_img_mcan_embedding = hist_img_mcan_embedding * mask_

    # for batch in range(batch_size):
    #     for i in range(num_rounds):
    #         hist_img_mcan_embedding[batch*num_rounds + i, i+1:num_rounds, :] = 0
    #
    # print(hist_img_mcan_embedding)


    masked_v = mask2d(
                        torch.ones((num_rounds, num_rounds, emb_size)),
                        torch.arange(1, num_rounds + 1).int(),
                        v_mask=0)
    # print(masked_v)
    mask = masked_v.unsqueeze(0).expand(batch_size, num_rounds, num_rounds)


def mask2d(value, sizes, v_mask=0, v_unmask=1):
    """Mask entries in value with `v_mask` based on sizes.
    Args
    ----
    value: Tensor of size (B, N)
        Tensor to be masked.
    sizes: list of int
        List giving the number of valid values for each item
        in the batch. Positions beyond each size will be masked.
    Returns
    -------
    value:
        Masked value.
    """
    mask = value.data.new(value.size()).fill_(v_unmask)
    n = mask.size()

    for i in range(num_rounds):
        value[batch*num_rounds + i, i+1:num_rounds, :] = 0


    return value



def test_mask():
    masking_try()


if __name__ == "__main__":
    test_mask()
