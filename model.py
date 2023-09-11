import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # Implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(384, 128, 3, padding = 1)
        self.conv7 = nn.Conv2d(192, 64, 3, padding = 1)
        self.conv8 = nn.Conv2d(96, 32, 3, padding = 1)
        self.conv9 = nn.Conv2d(48, 16, 3, padding = 1)
        self.conv10 = nn.Conv2d(16, 6, 1)
        

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        
        """
        # ReLu
        
        # Block 1
        x = F.relu(self.conv1(x))
        x1 = x.clone()
        x = F.max_pool2d(x, 2)
        
        # Block 2
        x = F.relu(self.conv2(x))
        x2 = x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        # Block 3
        x = F.relu(self.conv3(x))
        x3 = x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        # Block 4
        x = F.relu(self.conv4(x))
        x4 = x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        # Block 5
        x = F.interpolate(F.relu(self.conv5(x)), scale_factor = 2)
        x = torch.cat((x4, x), dim = 1)
        
        # Block 6
        x = F.interpolate(F.relu(self.conv6(x)), scale_factor = 2)
        x = torch.cat((x3, x), dim = 1)
        
        # Block 7 
        x = F.interpolate(F.relu(self.conv7(x)), scale_factor = 2)
        x = torch.cat((x2, x), dim = 1)
        
        # Block 8
        x = F.interpolate(F.relu(self.conv8(x)), scale_factor = 2)
        x = torch.cat((x1, x), dim = 1)
        
        # Block 9
        x = F.relu(self.conv9(x))
        
        # Block 10
        output = F.relu(self.conv10(x))
        """
        
        #Leaky ReLu
        
        # Block 1
        x = F.leaky_relu(self.conv1(x))
        x1 = x.clone()
        x = F.max_pool2d(x, 2)
        
        # Block 2
        x = F.leaky_relu(self.conv2(x))
        x2 = x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        # Block 3
        x = F.leaky_relu(self.conv3(x))
        x3 = x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        # Block 4
        x = F.leaky_relu(self.conv4(x))
        x4 = x.clone()
        x = F.max_pool2d(x, (2, 2))
        
        # Block 5
        x = F.interpolate(F.leaky_relu(self.conv5(x)), scale_factor = 2)
        x = torch.cat((x4, x), dim = 1)
        
        # Block 6
        x = F.interpolate(F.leaky_relu(self.conv6(x)), scale_factor = 2)
        x = torch.cat((x3, x), dim = 1)
        
        # Block 7 
        x = F.interpolate(F.leaky_relu(self.conv7(x)), scale_factor = 2)
        x = torch.cat((x2, x), dim = 1)
        
        # Block 8
        x = F.interpolate(F.leaky_relu(self.conv8(x)), scale_factor = 2)
        x = torch.cat((x1, x), dim = 1)
        
        # Block 9
        x = F.leaky_relu(self.conv9(x))
        
        # Block 10
        output = F.leaky_relu(self.conv10(x)) 

        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)