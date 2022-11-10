import time

import tqdm

loss = 0.9
bar = tqdm.tqdm(range(100), bar_format='{desc}|{bar}|{percentage:3.0f}%')
for i in bar:
    loss -= 0.001
    bar.set_description('epoch: [{}/{}] loss: {}'.format(i + 1, 100, loss))
    time.sleep(1)