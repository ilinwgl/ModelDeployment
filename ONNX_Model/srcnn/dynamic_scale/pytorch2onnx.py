import torch
import torch.onnx
from srcnn import StrangeSuperResolutionNet

x = torch.randn(1, 3, 256, 256) 
factor = torch.tensor([1, 1, 3, 3], dtype=torch.float) 

torch_model = StrangeSuperResolutionNet() 
state_dict = torch.load('../srcnn.pth')['state_dict'] 
 
# Adapt the checkpoint 
for old_key in list(state_dict.keys()): 
    new_key = '.'.join(old_key.split('.')[1:]) 
    state_dict[new_key] = state_dict.pop(old_key) 
 
torch_model.load_state_dict(state_dict) 
torch_model.eval() 

with torch.no_grad(): 
    torch.onnx.export(torch_model, (x, factor), 
                      "srcnn.onnx", 
                      opset_version=11, 
                      input_names=['input', 'factor'], 
                      output_names=['output']) 