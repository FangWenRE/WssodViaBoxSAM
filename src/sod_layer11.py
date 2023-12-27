import torch
import torch.nn as nn
import torch.nn.functional as F
from src.resnet import ResNet101, ResNet50
from src.component import ASPP, AttentionDecoder

class Deeplabv3plus(nn.Module):
	def __init__(self, os):
		super(Deeplabv3plus, self).__init__()
		self.os = os
		if os == 16:
			rates = [1, 6, 12, 18]
		elif os == 8 or os == 32:
			rates = [1, 12, 24, 36]
		else:
			raise NotImplementedError

		# 特征提取
		self.backbone_features = ResNet101(os)
		self.aspp = ASPP(2048, 256, rates)

		# 低特征融合
		self.ll_feats1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64)) 
		self.ll_feats2 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64))

		# 融合边缘
		self.sal_cat = nn.Sequential(nn.Conv2d(256+64, 256, 3, stride=1, padding=1, bias=False),
									nn.BatchNorm2d(256), nn.ReLU()) 
		
		# 特征图解码器
		self.sal_pred = nn.Sequential(nn.Conv2d(256+64, 256, 3, stride=1, padding=1, bias=False),
									nn.BatchNorm2d(256), nn.ReLU(),
									nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
									nn.BatchNorm2d(256), nn.ReLU(),
									nn.Conv2d(256, 1, 1, stride=1))

		self.attention1 = AttentionDecoder(256)

		self._init_weight()

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)

		weight =  "/opt/checkpoint/resnet101.pth"
		pretrain_dict = torch.load(weight)
		model_dict = {}
		state_dict = self.backbone_features.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict: model_dict[k] = v
		state_dict.update(model_dict)
		self.backbone_features.load_state_dict(state_dict)
		print(">>> init_weight")

	def _upsample_add(self, x, y):
		_, _, H, W = y.size()
		return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

	def forward(self, input):
		# Salient
		x0, x1, x2, x3, x4 = self.backbone_features(input) 

		if self.os == 32:
			x4 = F.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=True)

		aspp_feat = self.aspp(x4)
		
		# low_level
		low_level_features = self.ll_feats1(x1) # 128，128，265
		up_size = low_level_features.size()[2:]

		x2_level_features = self.ll_feats2(x1)

		aspp_feat = F.interpolate(aspp_feat, up_size, mode='bilinear', align_corners=True)
		high_low_feats = self.sal_cat(torch.cat([aspp_feat, low_level_features], dim=1))

		sal_atten = self.attention1(high_low_feats, high_low_feats)
		sal_out = torch.cat([sal_atten, x2_level_features], dim=1)
		sal_pred = F.interpolate(self.sal_pred(sal_out), input.size()[2:], mode='bilinear', align_corners=True)
		
		return sal_pred
	
if __name__ == '__main__':
    dummy_input = torch.randn(2, 3, 512, 512)

    model = Deeplabv3plus(16)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    out = model(dummy_input)
    print(out.shape)
