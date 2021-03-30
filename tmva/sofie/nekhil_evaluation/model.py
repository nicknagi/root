import torch.nn as nn
import torch.nn.init as init
import torch

class simpleNN(nn.Module):
    def __init__(self, inplace=False):
        super(simpleNN, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x, y):
        x = self.relu(x+y)
        return x

torch_model = simpleNN()
torch_model.train(False)

x = torch.randn(10,10)
y = torch.randn(10,10)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  (x,y),                         # model input (or a tuple for multiple inputs)
                  "simpleNN.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
