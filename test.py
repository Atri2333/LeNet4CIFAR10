import os
import torch
from PIL import Image
from torchvision import transforms
from model import *

lenet = LeNet()
#choose one model from 100(=epoch) models
lenet.load_state_dict(torch.load("model/Trained_Lenet81.pth"))
lenet.eval()

trans = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

idx_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
file_path = "img"
img_name = "1.jpg"

path = os.path.join(file_path, img_name)
img = Image.open(path)
img_tensor = trans(img)
img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))

with torch.no_grad():
    output = lenet(img_tensor)
    #print(output)
    print(idx_to_class[torch.argmax(output).item()])
