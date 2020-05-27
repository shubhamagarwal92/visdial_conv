class Cfgs():
    def __init__(self):
        super().__init__()

        self.LAYER = 6
        self.HIDDEN_SIZE = 512  # lstm_hidden_size
        self.BBOXFEAT_EMB_SIZE = 2048
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.USE_GLOVE = False
        # self.WORD_EMBED_SIZE = 20# 100, 6, 2
