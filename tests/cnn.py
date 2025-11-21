from vision_bench.models import CNN
import torch

model = CNN('resnet50')
output = model.run(torch.randn(2, 3, 224, 224))
#print(output.shape)
