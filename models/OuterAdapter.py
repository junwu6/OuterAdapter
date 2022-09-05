import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Function
import numpy as np
import torch.nn.functional as F


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class OuterAdapterModel(nn.Module):
    def __init__(self, num_classes, num_sources, option='resnet50', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(OuterAdapterModel, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
            self.dim = 512
        elif option == 'resnet34':
            model_ft = models.resnet34(pretrained=True)
            self.dim = 512
        elif option == 'resnet50':
            model_ft = models.resnet50(pretrained=True)
        elif option == 'resnet101':
            model_ft = models.resnet101(pretrained=True)
        elif option == 'resnet152':
            model_ft = models.resnet152(pretrained=True)
        elif option == 'vgg19':
            model_ft = models.vgg19(pretrained=True)
            self.dim = 25088

        mod = list(model_ft.children())
        mod.pop()
        self.base_network = nn.Sequential(*mod)
        self.use_bottleneck = use_bottleneck
        self.num_classes = num_classes
        self.num_sources = num_sources

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(self.dim, bottleneck_width),
            nn.BatchNorm1d(bottleneck_width),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        hidden_dim = bottleneck_width if self.use_bottleneck else self.dim
        self.classifier_layer = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )

        discriminator = nn.Sequential(
            nn.Linear(hidden_dim + num_classes, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, 2),
        )
        self.discriminator = nn.ModuleList([discriminator for _ in range(num_sources)])

        discriminator2 = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, 2),
        )
        self.discriminator2 = nn.ModuleList([discriminator2 for _ in range(num_sources)])
        self.attention_layer = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_classes)])

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, s_inputs, s_outputs, t_inputs, alpha, mode='auto'):
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            t_feats = self.bottleneck_layer(t_feats)
        t_preds = self.classifier_layer(t_feats)
        t_preds = F.softmax(t_preds, dim=1)

        all_loss = []
        all_sfeats = []
        for i in range(len(s_inputs)):
            s_feats = self.base_network(s_inputs[i]).view(s_inputs[i].shape[0], -1)
            if self.use_bottleneck:
                s_feats = self.bottleneck_layer(s_feats)
            s_preds = self.classifier_layer(s_feats)
            all_sfeats.append(s_feats)
            class_loss = self.criterion(s_preds, s_outputs[i])

            # prior = 0.1
            prior = 0.0001 # office-31
            pu_loss, unk_loss = 0, 0
            t_rev_preds = self.classifier_layer(ReverseLayerF.apply(t_feats, alpha))
            s_rev_preds = self.classifier_layer(ReverseLayerF.apply(s_feats, alpha))
            for c in range(self.num_classes):
                funk_labels = np.array([c] * s_inputs[0].shape[0])
                target_funk = torch.tensor(funk_labels, requires_grad=False, dtype=torch.long, device=t_inputs.device)
                if c == self.num_classes - 1:
                    unk_loss += self.criterion(t_preds, target_funk) - self.criterion(s_preds, target_funk) * prior
                else:
                    pu_loss += self.criterion(t_rev_preds, target_funk) - self.criterion(s_rev_preds, target_funk) * prior

            num_1 = 12 - len(s_inputs)
            num_2 = 1 + len(s_inputs)
            p_prob = torch.sum(t_preds[:, :self.num_classes - 1], 1).view(-1, 1)
            idx = torch.argsort(p_prob.flatten(), descending=True)

            domain_preds = self.discriminator[i](ReverseLayerF.apply(torch.cat([torch.cat([s_feats, self.one_hot(s_outputs[i])], dim=1),
                                                                                torch.cat([t_feats[idx[:num_1]], t_preds[idx[:num_1]]], dim=1)], dim=0), alpha))
            domain_labels = np.array([0] * s_feats.shape[0] + [1] * num_1)
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=t_inputs.device)
            domain_loss = self.criterion(domain_preds, domain_labels)

            domain_preds = self.discriminator2[i](torch.cat([s_feats, t_feats[idx[-num_2:]]], dim=0))
            domain_labels = np.array([0] * s_feats.shape[0] + [1] * num_2)
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=t_inputs.device)
            domain_loss += 0.1 * self.criterion(domain_preds, domain_labels)
            all_loss.append(class_loss + 0.1*pu_loss / (self.num_classes-1) + 1.8 * unk_loss + domain_loss)

        all_loss = torch.stack(all_loss)
        loss = 0.
        if mode == 'mu':
            mu = 1.
            for i in range(len(s_inputs)):
                loss += (mu ** (len(s_inputs) - i - 1)) * all_loss[i]
        elif mode == 'auto':
            weight = self.estimate_alpha(all_sfeats, s_outputs)
            loss = torch.dot(all_loss, weight) + torch.norm(weight)
        return loss

    def one_hot(self, x):
        out = torch.zeros(len(x), self.num_classes).to(x.device)
        out[torch.arange(len(x)), x.squeeze()] = 1
        return out

    def estimate_alpha(self, sfeats, pesudo_soutputs):
        score = torch.zeros(size=[self.num_classes, self.num_sources], dtype=torch.float32, device=sfeats[0].device)
        for i in range(self.num_classes - 1):
            for j in range(self.num_sources):
                c_sfeats = torch.mean(sfeats[j][pesudo_soutputs[j] == i, :], dim=0, keepdim=True)
                c_sfeats[torch.isnan(c_sfeats)] = 0.0
                score[i, j] = self.attention_layer[i](c_sfeats)
        score = torch.sum(score, dim=0)
        score = torch.exp(F.leaky_relu(score))
        score = score / torch.sum(score)
        return score

    def inference(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return self.classifier_layer(x)
