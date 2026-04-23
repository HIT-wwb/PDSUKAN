import PDSUKAN
import torch
from train import parse_args
from fvcore.nn import FlopCountAnalysis, parameter_count_table
config = vars(parse_args())
model = PDSUKAN.__dict__[config['arch6.9']](
    num_classes=config['num_classes'],
    input_channels=config['input_channels'],
    deep_supervision=config['deep_supervision'],
    embed_dims=config['input_list']
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
dummy_input = torch.randn(1, 3, 256, 256).to(device)
flops = FlopCountAnalysis(model, dummy_input)
print(f"FLOPs: {flops.total() / 1e9} G")
print(parameter_count_table(model))