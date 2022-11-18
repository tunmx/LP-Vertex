import cv2
from breezevertex.data import LabelMeDataset
from torch.utils.data import DataLoader
from breezevertex.utils.image_tools import visual_images

if __name__ == '__main__':
    batch_size = 32
    path_list = ['oinbagCrawler_vertex_train/val', '/Users/tunm/Downloads/ccpd_all_vertex/data']
    train_dataset = LabelMeDataset(path_list, mode='train', is_show=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    img_tensor, label_tensor = next(iter(train_dataloader))
    print(img_tensor.shape)
    print('datasets: ', len(train_dataset))
    draw_list = visual_images(img_tensor, label_tensor, 112, 112)

    for img in draw_list:
        cv2.imshow("w", img)
        cv2.waitKey(0)

    # for idx, img in enumerate(img_tensor):
    #     print(img.dtype)
        # img = img.numpy()
        # kps = label_tensor[idx].numpy()
        # for x, y in kps.astype(np.int32):
        #     cv2.line(img, (x, y), (x, y), (100, 100, 255), 3)
        # cv2.imshow("w", img)
        # cv2.waitKey(0)