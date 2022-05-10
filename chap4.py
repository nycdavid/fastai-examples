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
# ax = show_image(mean3)
# ax.figure.savefig("mean3.png")
#
mean7 = stacked_sevens.mean(0)
# ax = show_image(mean7)
# ax.figure.savefig("mean7.png")

a_3 = stacked_threes[1]
dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()

a_7 = stacked_sevens[1]
dist_7_abs = (a_7 - mean7).abs().mean()
dist_7_sqr = ((a_7 mean7)**2).mean().sqrt()

valid_3_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()])
valid_3_tens = valid_3_tens.float() / 255

valid_7_tens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float() / 255

# mean absolute value
def mnist_distance(a, b):
  return (a - b).abs().mean()
