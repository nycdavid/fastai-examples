threes = []

result = torch.tensor([])
# stacking all threes
for path in threes:
  # open the path
  # convert to a PyTorch tensor
  img = Image.open(path)
  tensor = torch.tensor(array(img))
