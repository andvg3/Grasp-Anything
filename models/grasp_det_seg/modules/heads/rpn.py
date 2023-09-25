import torch.nn as nn

# from inplace_abn import ABN

class RPNHead(nn.Module):
    """RPN head module

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map
    num_anchors : int
        Number of anchors predicted at each spatial location
    stride : int
        Stride of the internal convolutions
    hidden_channels : int
        Number of channels in the internal intermediate feature map
    norm_act : callable
        Function to create normalization + activation modules
    """

    def __init__(self, in_channels, num_anchors, stride=1, hidden_channels=255, norm_act=nn.BatchNorm2d):
        super(RPNHead, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv_obj = nn.Conv2d(hidden_channels, num_anchors, 1)
        self.conv_bbx = nn.Conv2d(hidden_channels, num_anchors * 4, 1)

        self.reset_parameters()

    def reset_parameters(self):
        activation = "relu"
        activation_param = self.bn1.weight

        # Hidden convolution
        gain = nn.init.calculate_gain(activation, activation_param)
        nn.init.xavier_normal_(self.conv1.weight, gain)
        self.bn1.reset_parameters()

        # Classifiers
        for m in [self.conv_obj, self.conv_bbx]:
            nn.init.xavier_normal_(m.weight, .01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """RPN head module
        """
        x = self.conv1(x)
        x = self.bn1(x)
        return self.conv_obj(x), self.conv_bbx(x)
