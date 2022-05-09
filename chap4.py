from fastai.vision.all import *
from fastbook import *

path = untar_data(URLs.MNIST_SAMPLE)
threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()

three_tensors = [tensor(Image.open(img_path)) for img_path in threes]
seven_tensors = [tensor(Image.open(img_path)) for img_path in sevens]

stacked_threes = torch.stack(three_tensors).float() / 255
stacked_sevens = torch.stack(seven_tensors).float() / 255

mean3 = stacked_threes.mean(0)
ax = show_image(mean3)
ax.figure.save_fig("mean3.png")

mean7 = stacked_sevens.mean(0)
ax = show_image(mean7)
ax.figure.save_fig(mean7)
