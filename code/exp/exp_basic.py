import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, MPNN,TTT,TSATT,TSATT2,TSATT3,LinearRegressionModel,fiveTS3,zeroNO,zeroTN,zeroAu , \
    zeroTS3_sin1,fiveTS3_sin2,zeroNO_75,zeroAu_75,zeroTN_75,MPNN_75,zeroTS3_sin1_75,zeroTS3_era,zeroTS3_sta,zeroTS3_pos_only,zeroTS3_feat_only,zeroTS3_zero_embed ,\
    OK,Kriging,Kriging_75,zeroTF3_sin1


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'TTT':TTT,
            'TSATT':TSATT,
            'TSATT2':TSATT2,
            'TSATT3': TSATT3,
            'fiveTS3_sin2': fiveTS3_sin2,
            'fiveTS3': fiveTS3,
            'LR':LinearRegressionModel,#baseline
            'zeroNO':zeroNO,
            'zeroTN':zeroTN,
            'zeroAu':zeroAu,
            'MPNN': MPNN,
            'zeroNO_75': zeroNO_75,
            'zeroAu_75':zeroAu_75,
            'zeroTN_75':zeroTN_75,
            'MPNN_75':MPNN_75,
            'zeroTS3_sin1': zeroTS3_sin1,
            'zeroTF3_sin1': zeroTF3_sin1,
            'zeroTS3_sin1_75': zeroTS3_sin1_75,
            'zeroTS3_era':zeroTS3_era,
            'zeroTS3_sta':zeroTS3_sta,
            'zeroTS3_pos_only':zeroTS3_pos_only,
            'zeroTS3_feat_only':zeroTS3_feat_only,
            'zeroTS3_zero_embed':zeroTS3_zero_embed,
            'ok':OK,
            'Kriging':Kriging,
            'Kriging_75':Kriging_75,

        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
