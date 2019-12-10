
import torch

##################################################################################
#                            Creating of the Model                               #
##################################################################################
class ConvNetDoS(torch.nn.Module):

    def __init__(self):
        """ 

        Initialization of the CNN module 
        
        Args:

        """


        super(ConvNetDoS, self).__init__()                  # Super class init called

        # Define the first convolution layer 
        self._layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 14, kernel_size=3, stride=1, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )

        # Define the seconde convolution layer

        self._layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(14, 28, kernel_size=3, stride=1, padding=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )

        # Flat the output of the last convolution layer
        self._dropout = torch.nn.Dropout2d()
        
        # Creating of the full connected layers (FCs)
        self._input      = 12 * 12 * 28
        self._output     = 12 * 12 * 28 * 2 + 1
        self._class_num  = 5
        # First FC (in_features = self._input, out_features = self._output)
        self._fc1        = torch.nn.Linear(self._input, self._output)

        # Second FC (in_features=self._output, out_features=self._class_num)
        self._fc2        = torch.nn.Linear(self._output, self._class_num)

    def forward(self, matrix_in):
        output = self._layer1(matrix_in)
        #print(output)
        output = self._layer2(output)
        output = output.reshape(output.size(0), -1)
        output = self._dropout(output)
        output = self._fc1(output)
        output = self._fc2(output)
        
        return output

# End of model definition

