from data import VertexDataset
from model.shuffle_vertex import ShuffleVertex
from torch.utils.data import DataLoader
import torch

device = torch.device("cpu")
net = ShuffleVertex()
net.load_state_dict(torch.load("save_dir/best_model.pth", map_location="cpu"))
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