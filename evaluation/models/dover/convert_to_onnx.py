import torch
import torch.nn as nn

import dover
from dover.models import VQAHead
from dover.models import VQABackbone as VideoBackbone, convnext_3d_tiny

class MinimumDOVER(nn.Module):
    def __init__(self):
        super().__init__()
        self.technical_backbone = VideoBackbone(use_checkpoint=False)
        self.aesthetic_backbone = convnext_3d_tiny(pretrained=False)
        self.technical_head = VQAHead(pre_pool=False, in_channels=768)
        self.aesthetic_head = VQAHead(pre_pool=False, in_channels=768)


    def forward(self,aesthetic_view, technical_view):
        self.eval()
        with torch.no_grad():
            aesthetic_score = self.aesthetic_head(self.aesthetic_backbone(aesthetic_view))
            technical_score = self.technical_head(self.technical_backbone(technical_view))
            
        aesthetic_score_pooled = torch.mean(aesthetic_score, (1,2,3,4))
        technical_score_pooled = torch.mean(technical_score, (1,2,3,4))
        return [aesthetic_score_pooled, technical_score_pooled]
    
import torch
minimum_dover = MinimumDOVER()
sd = torch.load("pretrained_weights/DOVER.pth", map_location="cpu")
minimum_dover.load_state_dict(sd)

if torch.cuda.is_available():
    minimum_dover = minimum_dover.cuda()
    dummy_inputs = (torch.randn(1,3,32,224,224).cuda(), torch.randn(4,3,32,224,224).cuda()) 
else:
    dummy_inputs = (torch.randn(1,3,32,224,224), torch.randn(4,3,32,224,224))
    
torch.onnx.export(minimum_dover, dummy_inputs, "onnx_dover.onnx", verbose=True, 
                  input_names=["aes_view", "tech_view"])

print("Successfull")