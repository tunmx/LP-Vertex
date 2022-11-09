import cv2
import numpy as np

from data import VertexDataset
from model.shuffle_vertex import ShuffleVertex
from torch.utils.data import DataLoader
import torch

device = torch.device("cpu")
net = ShuffleVertex()
net.load_state_dict(torch.load("save_dir/last.pth", map_location="cpu"))
net.to(device)

batch_size = 1
val_dir = "oinbagCrawler_vertex_train/val"
val_dataset = VertexDataset(val_dir, mode='val', is_show=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

img_tensor, label_tensor = next(iter(val_dataloader))

net.eval()
y = net(img_tensor)

print(label_tensor)
print(y)

image_decode = img_tensor[0].numpy() * 255
image_decode = image_decode.transpose(1, 2, 0)
image_decode = image_decode.astype(np.uint8)
image_decode = image_decode.copy()
h, w, _ = image_decode.shape
print(image_decode.shape)

kps = y.detach().numpy().reshape(4, 2)
kps[:, 0] *= w
kps[:, 1] *= h

for x, y in kps.astype(np.int32):
    print(x, y)
    cv2.line(image_decode, (x, y), (x, y), (100, 100, 255), 3)

cv2.imshow("w", image_decode)
cv2.waitKey(0)