import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch




class ResNetCHR(nn.Module):

    def __init__(self, model, num_classes, dense=True):
        super(ResNetCHR, self).__init__()

        #self.dense = dense
        for item in model.children():
            if isinstance(item,nn.BatchNorm2d):
                item.affine=False

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.cov4 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1)
        self.cov3 = nn.Conv2d(3072, 1024, kernel_size=1, stride=1)
        self.cov2 = nn.Conv2d(1536, 512, kernel_size=1, stride=1)

        self.cov3_1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.cov2_1 = nn.Conv2d(512, 512, kernel_size=1, stride=1)

        self.po1 = nn.AvgPool2d(7, stride=1)
        self.po2 = nn.AvgPool2d(14, stride=1)
        self.po3 = nn.AvgPool2d(28, stride=1)
        self.fc1 = nn.Linear(2048, num_classes)
        self.fc2 = nn.Linear(1024, num_classes)
        self.fc3 = nn.Linear(512, num_classes)



        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def _upsample_add(self,x,y):

        _, _, H, W=y.size()
        z = F.upsample(x, size=(H, W), mode='bilinear')
        return  torch.cat([z,y],1)
    def get_config_optim(self,lr,lrp):
        return [{'params':self.features.parameters()},
                {'params':self.layer1.parameters()},
                {'params':self.layer2.parameters()},
                {'params':self.layer3.parameters()},
                {'params':self.layer4.parameters()},
                {'params':self.cov4.parameters()},
                {'params':self.cov3.parameters()},
                {'params':self.cov2.parameters()},
                {'params':self.cov3_1.parameters()},
                {'params':self.cov2_1.parameters()},
                {'params':self.po1.parameters()},
                {'params':self.po2.parameters()},
                {'params':self.po3.parameters()},
                {'params':self.fc1.parameters()},
                {'params':self.fc2.parameters()},
                {'params':self.fc3.parameters()}]

    def forward(self, x):
        x = self.features(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l4_1 = self.cov4(l4)
        l4_2 = F.relu(l4_1)
        l4_3 = self.po1(l4_2)
        l4_4 = l4_3.view(l4_3.size(0), -1)
        o1 = self.fc1(l4_4)
        l3_1 = self.cov3_1(l3)
        l3_2 = F.relu(l3_1)
        l3_3 = self._upsample_add(l4,l3_2)
        l3_4 = self.cov3(l3_3)
        l3_5 = F.relu(l3_4)
        l3_6 = self.po2(l3_5)
        l3_7 = l3_6.view(l3_6.size(0), -1)
        o2 = self.fc2(l3_7)
        l2_1 =self.cov2_1(l2)
        l2_2 = F.relu(l2_1)
        l2_3 = self._upsample_add(l3_5,l2_2)
        l2_4 = self.cov2(l2_3)
        l2_5 = F.relu(l2_4)
        l2_6 = self.po3(l2_5)
        l2_7 = l2_6.view(l2_6.size(0), -1)
        o3 = self.fc3(l2_7)

        return o1,o2,o3







def resnet101_CHR(num_classes, pretrained=True):
    model = models.resnet101(pretrained)

    return ResNetCHR(model, num_classes )

