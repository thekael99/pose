import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from module import Extractor, Distance, ConvEncoder, FCEncoder, FC3DPose, FCDecoder, ConvDecoder


class Pose3D(nn.Module):
    def __init__(self):
        super(Pose3D, self).__init__()
        self.extractor = Extractor()
        self.distance = Distance()
        self.convEnc = ConvEncoder()
        self.fcEnc = FCEncoder()
        self.fc3DPose = FC3DPose()
        self.fcDec = FCDecoder()
        self.convDec = ConvDecoder()

    def forward(self, x):
        heatmap, features_dist = self.extractor(x)
        distances = self.distance(features_dist)
        conv_encoded = self.convEnc(heatmap)
        latent = self.fcEnc(conv_encoded)
        fc_decoded = self.fcDec(latent)
        heatmap_res = self.convDec(fc_decoded)
        pose = self.fc3DPose(latent, distances)
        return heatmap, heatmap_res, pose
