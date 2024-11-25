import timm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
# from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision


class Embedding(nn.Module):
    def __init__(self, is_train):
        super(Embedding, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)
        if is_train == True:
            modelpath = "../../support_data/checkpoints/resnet50-19c8e357.pth"
            self.resnet.load_state_dict(torch.load(modelpath))
        self.resnet.train()
        self.mlp = MLP(1000, 1024, 32)

    def forward(self, img):
        out = self.resnet(img)
        out = self.mlp(out)
        return out

class Embedding_Vit(nn.Module):
    def __init__(self, is_train):
        super(Embedding_Vit, self).__init__()
        
        self.vit = timm.create_model(
        'vit_huge_patch14_224.mae',
        pretrained=False,
        # checkpoint_path="./model/vit/pytorch_model.bin",
        num_classes=0,  # remove classifier nn.Linear
        )

        if is_train == True:
            self.vit.train()

        # self.data_config = timm.data.resolve_model_data_config(self.vit)
        # self.transforms = timm.data.create_transform(**self.data_config, is_training=False)

        # if is_train == True:
        #     modelpath = "./resnet50-19c8e357.pth"
        #     self.resnet.load_state_dict(torch.load(modelpath))
        # self.resnet.train()
        self.mlp = MLP(1280, 1024, 32)

    def forward(self, img):
        out = self.vit(img)
        # out = self.resnet(img)
        out = self.mlp(out)
        return out

class MLP(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(MLP, self).__init__()
        self.hidden0 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden1 = torch.nn.Linear(n_hidden, 512)
        self.hidden2 = torch.nn.Linear(512, 256)
        self.hidden3 = torch.nn.Linear(256, 128)
        self.predict = torch.nn.Linear(128, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden0(x))
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.predict(x)
        return x


if __name__ == "__main__":
    img_path = "beignets-task-guide.png"
    img = Image.open(img_path)
    pass
