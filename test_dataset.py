import cv2
import numpy as np
from data import LabelMeDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    batch_size = 32
    train_dataset = LabelMeDataset("oinbagCrawler_vertex_train/train", mode='train', is_show=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    img_tensor, label_tensor = next(iter(train_dataloader))
    print(img_tensor.shape)
    print('datasets: ', len(train_dataset))
    # for idx, img in enumerate(img_tensor):
    #     print(img.dtype)
        # img = img.numpy()
        # kps = label_tensor[idx].numpy()
        # for x, y in kps.astype(np.int32):
        #     cv2.line(img, (x, y), (x, y), (100, 100, 255), 3)
        # cv2.imshow("w", img)
        # cv2.waitKey(0)