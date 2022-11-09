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

batch_size = 128
val_dir = "oinbagCrawler_vertex_train/val"
val_dataset = VertexDataset(val_dir, mode='val', is_show=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

img_tensor, label_tensor = next(iter(val_dataloader))

net.eval()
with torch.no_grad():
    y_hat = net(img_tensor)

print(label_tensor)
print(y_hat)

y_hat = y_hat.detach().numpy()

for idx in range(batch_size):
    image_decode = img_tensor[idx].numpy() * 255
    image_decode = image_decode.transpose(1, 2, 0)
    image_decode = image_decode.astype(np.uint8)
    image_decode = image_decode.copy()
    h, w, _ = image_decode.shape
    print(image_decode.shape)
    print(label_tensor[idx].numpy())
    print(y_hat[idx])


    kps = y_hat[idx].reshape(4, 2)
    kps[:, 0] *= w
    kps[:, 1] *= h

    for x, y in kps.astype(np.int32):
        # print(x, y)
        cv2.line(image_decode, (x, y), (x, y), (100, 100, 255), 3)

    cv2.imshow("w", image_decode)
    cv2.waitKey(0)