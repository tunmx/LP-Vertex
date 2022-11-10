import cv2
import numpy as np

from data import LabelMeDataset
from model.shuffle_vertex import ShuffleVertex
from torch.utils.data import DataLoader
import torch

device = torch.device("cpu")
net = ShuffleVertex()
net.load_state_dict(torch.load("save_dir/last.pth", map_location="cpu"))
net.to(device)

batch_size = 128
val_dir = "oinbagCrawler_vertex_train/val"
val_dataset = LabelMeDataset(val_dir, mode='val', is_show=False)
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

    pts = np.float32([[0, 0], [w * 2, 0], [w * 2, h], [0, h], ])
    matrix = cv2.getPerspectiveTransform(kps, pts)
    output = cv2.warpPerspective(image_decode, matrix, (w * 2, h))



    for x, y in kps.astype(np.int32):
        # print(x, y)
        cv2.line(image_decode, (x, y), (x, y), (100, 100, 255), 3)



    x2 = cv2.resize(output, (112, 112))
    data2 = x2 / 255.0
    data2 = data2.transpose(2, 0, 1)
    data2 = np.expand_dims(data2, 0)
    data2 = torch.Tensor(data2)
    kps2 = net(data2).detach().numpy()
    kps2 = kps2.reshape(4, 2)
    kps2[:, 0] *= w
    kps2[:, 1] *= h


    for x, y in kps2.astype(np.int32):
        # print(x, y)
        cv2.line(x2, (x, y), (x, y), (100, 100, 255), 3)

    cv2.imshow("q", output)
    cv2.imshow("w", image_decode)
    cv2.imshow("x2", x2)
    cv2.waitKey(0)
