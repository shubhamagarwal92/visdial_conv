from visdialch.encoders.lf import LateFusionEncoder
from visdialch.encoders.mcan_img_only import MCANImgOnlyEncoder
from visdialch.encoders.mcan_img_mcan_vqa_hist_attn import MCANImgMCANVQAHistAttnEncoder
from visdialch.encoders.mcan_img_mcan_hist import MCANImgMCANHistEncoder
from visdialch.encoders.mcan_hist_only import MCANHistOnlyEncoder

def Encoder(model_config, *args):
    name_enc_map = {"lf": LateFusionEncoder,
                    "mcan_img_mcan_hist":MCANImgMCANHistEncoder,
                    "mcan_hist_only": MCANHistOnlyEncoder,
                    "mcan_img_mcan_vqa_hist_attn": MCANImgMCANVQAHistAttnEncoder,
                    "mcan_img_only": MCANImgOnlyEncoder}
    return name_enc_map[model_config["encoder"]](model_config, *args)
