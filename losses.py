"""
Description
++++++++++++++++++++++
Addition losses module defines classses which are commonly used particularly in segmentation and are not part of standard pytorch library.

Usage
++++++++++++++++++++++
Import the package and Instantiate any loss class you want to you::

    from nn_common_modules import losses as additional_losses
    loss = additional_losses.DiceLoss()

Members
++++++++++++++++++++++
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.autograd import Variable


class DiceLoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, ignore_index=None, binary=False):
        """
        Forward pass

        :param output: NxCxHxW logits
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return: torch.tensor
        """
        #output = F.softmax(output, dim=1)
        if binary:
            return self._dice_loss_binary(output, target, weights)
        return self._dice_loss_multichannel(output, target, weights, ignore_index)

    @staticmethod
    def _dice_loss_binary(output, target, weights=None):
        """
        Dice loss for one channel binarized input

        :param output: Nx1xHxW logits
        :param target: NxHxW LongTensor
        :return:
        """

        smooth = 1.
        target = target.float()
        intersection = output * target
        numerator = 2 * (intersection.sum())
        denominator = output + target
        denominator = denominator.sum()
        loss_per_channel = 1 - ((numerator + smooth) / (denominator + smooth))

        return loss_per_channel

    @staticmethod
    def _dice_loss_multichannel(output, target, weights=None, ignore_index=None):
        """
        Forward pass

        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return:
        """
        eps = 0.0001
        encoded_target = output.detach() * 0
        target = target.squeeze()
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class IoULoss(_WeightedLoss):
    """
    IoU Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, ignore_index=None):
        """Forward pass
        
        :param output: shape = NxCxHxW
        :type output: torch.tensor [FloatTensor]
        :param target: shape = NxHxW
        :type target: torch.tensor [LongTensor]
        :param weights: shape = C, defaults to None
        :type weights: torch.tensor [FloatTensor], optional
        :param ignore_index: index to ignore from loss, defaults to None
        :type ignore_index: int, optional
        :return: loss value
        :rtype: torch.tensor
        """

        output = F.softmax(output, dim=1)

        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = intersection.sum(0).sum(1).sum(1)
        denominator = (output + encoded_target) - (output*encoded_target)

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class BinaryCrossEntropy2D(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self):
        super(BinaryCrossEntropy2D, self).__init__()
        self.nll_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        """
        Forward pass
        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(inputs, targets)



class CombinedLoss(_Loss):
    """
    A combination of dice and cross entropy loss
    """

    def __init__(self, weight_mfb=None, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.cross_entropy = BinaryCrossEntropy2D()

    def forward(self, input, target, class_weight=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        y_1, y_2 = self.calc_losses(input, target, class_weight=class_weight)
        alpha = 1
        beta = 1
        return alpha*y_1 + beta*y_2


    def calc_losses(self, output, target, class_weight=None):

        y_1 = self.dice_loss(output, target, binary=True)
        y_2 = torch.mean(torch.mul(self.cross_entropy.forward(output, target.float()), class_weight))

        # y_2 = F.binary_cross_entropy(output, target.float(), weight=class_weight, reduction='mean')
        #y_2 = torch.mean(torch.mul(y_2, class_weight.cuda()))
        return y_1, y_2

