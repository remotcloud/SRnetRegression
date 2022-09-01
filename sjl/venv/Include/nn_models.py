from torch import nn


class MLP2(nn.Module):
    def __init__(self, num_input, num_output, n_hidden):
        super(MLP2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_input, n_hidden),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid()
        )
        self.fc3 = nn.Linear(n_hidden, num_output)

    def forward(self, x):
        output1 = self.fc1(x)
        output2 = self.fc2(output1)
        outputs = self.fc3(output2)
        return output1, output2, outputs


class OlderMLP2(nn.Module):
    def __init__(self, num_input, num_output, n_hidden):
        super(OlderMLP2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_input, n_hidden),
            nn.Tanh()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh()
        )
        self.fc3 = nn.Linear(n_hidden, num_output)

    def forward(self, x):
        output1 = self.fc1(x)
        output2 = self.fc2(output1)
        outputs = self.fc3(output2)
        return output1, output2, outputs


class MLP3(nn.Module):
    def __init__(self, num_input, num_output, n_hidden):
        super(MLP3, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(num_input, n_hidden),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Sigmoid()
        )
        self.fc4 = nn.Linear(n_hidden, num_output)

    def forward(self, x):
        output1 = self.fc1(x)
        output2 = self.fc2(output1)
        output3 = self.fc3(output2)
        outputs = self.fc4(output3)
        return output1, output2, output3, outputs
class Activation_Net(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2,n_hidden_3,n_hidden_end,out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_end),nn.Sigmoid())
        # self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),nn.Sigmoid())
        # self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_end),nn.Sigmoid())
        self.layer_end = nn.Sequential(nn.Linear(n_hidden_end, out_dim))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x3 = self.layer_end(x2)
        return [x1,x2,x3]



NN_MAP = {
    'kkk0': MLP2(1, 1, 3),
    'kkk1': MLP2(2, 1, 3),
    'kkk2': MLP2(1, 1, 5),
    'kkk3': MLP3(2, 1, 4),
    'kkk4': MLP2(3, 1, 4),
    'kkk5': MLP2(2, 1, 5),
    'feynman0': MLP2(3, 1, 3),
    'feynman1': MLP2(4, 1, 3),
    'feynman2': MLP2(5, 1, 3),
    'feynman3': MLP2(2, 1, 3),
    'feynman4': MLP2(5, 1, 5),
    'feynman5': MLP3(5, 1, 5),
    'feynman7':Activation_Net(6,5,10,20,10,1)
}
