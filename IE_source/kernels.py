from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


from IE_source.Galerkin_transformer import SimpleTransformerEncoderLayer, SimpleTransformerEncoderLastLayer
from IE_source.Attentional_IE_solver import interval_function

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)

def linear_kernel(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat((x-s.reshape(s.size()[0],1,1),x+1-s.reshape(s.size()[0],1,1)),-1)
    B = torch.cat((x+1-s.reshape(s.size()[0],1,1),x-s.reshape(s.size()[0],1,1)),-1)
    C=torch.stack((A,B),-2)
    return C.reshape(s.size(0),2,2)  

def exp_kernel(x,s):
    A = torch.Tensor([torch.exp(-x),torch.tensor([0])])
    B = torch.Tensor([torch.tensor([0]),torch.exp(-x)])
    C=torch.stack((A,B),-2)
    return C.repeat(s.size(0),1,1)

def exp_kernel2(x,s):
    A = torch.cat([torch.exp(x/(1+s)).reshape(s.size(0),1),torch.zeros(s.size(0),1)],-1).to(device)
    B = torch.cat([torch.zeros(s.size(0),1),torch.exp(x/(1+s)).reshape(s.size(0),1)],-1).to(device)
    C = torch.stack([A,B],-1).to(device)
    return C

def exp_kernel3(x,s):
    A = torch.cat([torch.exp(x-s).reshape(s.size(0),1),torch.zeros(s.size(0),1)],-1).to(device)
    B = torch.cat([torch.zeros(s.size(0),1),torch.exp(x-s).reshape(s.size(0),1)],-1).to(device)
    C = torch.stack([A,B],-1).to(device)
    return C

def identity_kernel(x,s,dim):
    A = torch.ones(dim).to(device)
    B = torch.diag(A)
    return B.repeat(s.size(0),1,1)

def diagonal_cubic(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat((x**3-(s**3).reshape(s.size(0),1,1),torch.ones(s.size(0),1,1).to(device)),-1)
    B = torch.cat((torch.ones(s.size(0),1,1).to(device),x**3-(s**3).reshape(s.size(0),1,1)),-1)
    C=torch.stack((A,B),-2)
    return C.reshape(s.size(0),2,2)  

def cos_kernel(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat([torch.cos(x-s).reshape(s.size(0),1),-torch.sin(x-s).reshape(s.size(0),1)],-1)
    B = torch.cat([-torch.sin(x-s).reshape(s.size(0),1),-torch.cos(x-s).reshape(s.size(0),1)],-1)
    C = torch.stack([A,B],-1)
    return C

def sin_kernel(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat([torch.sin(x-s).reshape(s.size(0),1),+torch.cos(x-s).reshape(s.size(0),1)],-1)
    B = torch.cat([torch.cos(x-s).reshape(s.size(0),1),-torch.sin(x-s).reshape(s.size(0),1)],-1)
    C = torch.stack([A,B],-1)
    return C

def tanh_kernel(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat([torch.tanh(x-s).reshape(s.size(0),1),1-torch.sinh(x-s).reshape(s.size(0),1)],-1)
    B = torch.cat([1-torch.cosh(x-s).reshape(s.size(0),1),-torch.tanh(x-s).reshape(s.size(0),1)],-1)
    C = torch.stack([A,B],-1)
    return C


class neural_kernel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        
        #Give a tuple with several hidden dim options (future work)

        
        super(neural_kernel, self).__init__()

        self.lin1 = nn.Linear(in_dim+2, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim,hid_dim)
        self.lin4 = nn.Linear(hid_dim, hid_dim)
        self.lin5 = nn.Linear(hid_dim, hid_dim)
        self.lin6 = nn.Linear(hid_dim, hid_dim)
        self.lin7 = nn.Linear(hid_dim, hid_dim)
        self.lin8 = nn.Linear(hid_dim, out_dim)
        self.ELU = nn.ELU(inplace=True)

    def forward(self,y, x, t):
        y_in = torch.cat([y,x,t],-1)
        h = self.ELU(self.lin1(y_in))
        h = self.ELU(self.lin2(h))
        h = self.ELU(self.lin3(h))
        h = self.ELU(self.lin4(h))
        h = self.ELU(self.lin5(h))
        h = self.ELU(self.lin6(h))
        h = self.ELU(self.lin7(h))
        out = self.lin8(h)
        
        return out

class kernel_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(kernel_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim+2,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t, s):
        y_in = torch.cat([y,t,s],-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)

        return y

class G_global(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(G_global, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim+2,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t, s):
        y = y.squeeze()
        y_in = torch.cat([y,t,s],-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)

        return y
    

    
class F_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(F_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
     
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)

        return y
    

class model_blocks(nn.Module):
    def __init__(self,dimension,dim_emb,n_head, n_blocks,n_ff, attention_type, dim_out=2, Final_block = False,dropout=0.1,lower_bound=None,upper_bound=None):
        super(model_blocks, self).__init__()
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.first = nn.Linear(dimension+1,dim_emb)
        self.blocks = nn.ModuleList([SimpleTransformerEncoderLayer(
                                 d_model=dim_emb,n_head=n_head,
                                 dim_feedforward=n_ff,
                                 attention_type=attention_type,
                                 dropout=dropout) for i in range(n_blocks)])
        self.Final_block = Final_block
        if self.Final_block is True:
            self.last_block = SimpleTransformerEncoderLastLayer(
                                    d_model=dim_emb,n_head=n_head,
                                    dim_out=dim_out,dim_feedforward=n_ff,
                                    attention_type=attention_type,
                                    dropout=dropout)
        else:
            self.last_block = nn.Linear(dim_emb,dimension+1)#SimpleTransformerEncoderLayer(d_model=dim_emb,n_head=n_head,attention_type=attention_type,dim_feedforward=n_ff)
        
    def forward(self, x, dynamical_mask=None):
        
        x = self.first.forward(x)
        for block in self.blocks:
            x = block.forward(x,dynamical_mask=dynamical_mask) 
        if self.Final_block is True:
            x = self.last_block.forward(x,dynamical_mask=dynamical_mask)
        else:
            x = self.last_block.forward(x)

        return x

    
class Simple_NN(nn.Module):
    def __init__(self, in_dim, hid_dim,out_dim):
        super(Simple_NN, self).__init__()

        self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, hid_dim)
        self.lin4 = nn.Linear(hid_dim, out_dim)
        self.ELU = nn.ELU(inplace=True)
        
        self.in_dim = in_dim

    def forward(self,x,y):
        y = y.reshape(self.in_dim).to(device)
        x = x.reshape(1).to(device)
        
        y_in = torch.cat([x,y],-1)
        h = self.ELU(self.lin1(y_in))
        h = self.ELU(self.lin2(h))
        h = self.ELU(self.lin3(h))
        out = self.lin4(h)
        
        return out
    
    
class linear_nn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linear_nn, self).__init__()
        
        self.lin = nn.Linear(in_dim,out_dim,1)
        
    def forward(self,x):
        return self.lin(x)
    
def flatten_kernel_parameters(kernel):
    p_shapes = []
    flat_parameters = []
    for p in kernel.parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)


def flatten_F_parameters(NN_F):
    p_shapes = []
    flat_parameters = []
    for p in NN_F.parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)


def flatten_ode_parameters(ode_func):
    p_shapes = []
    flat_parameters = []
    for p in ode_func.parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)


##From Neural ODE paper https://arxiv.org/abs/1806.07366
class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)
    
class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
        
def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class Proj_enc(nn.Module):
    def __init__(self, in_dim, hid_dim,out_dim):
        super(Proj_enc, self).__init__()

        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.ELU = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.in_dim = in_dim

    def forward(self,y):
        y_in = y.to(device)
        
        h = self.ELU(self.lin1(y_in))
        h = self.lin2(h)
        out = self.sigmoid(h)
        
        return out
    

class Proj_dec(nn.Module):
    def __init__(self, in_dim, hid_dim,out_dim):
        super(Proj_dec, self).__init__()

        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.ELU = nn.ELU(inplace=True)
        
        self.in_dim = in_dim

    def forward(self,y):
        y_in = y.to(device)
        
        h = self.ELU(self.lin1(y_in))
        out = self.lin2(h)
        
        return out
    
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=64,Data_shape1=64,Data_shape2=64,n_patch=8):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(dim, hidden_dim,
                                     kernel_size=[int(Data_shape1/n_patch/2),int(Data_shape1/n_patch/2)],
                                    stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        if int(Data_shape1/n_patch/4) > 1:
            self.conv_layer2 = nn.Conv2d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/4),int(Data_shape1/n_patch/4)],
                                         stride=int(Data_shape1/n_patch/4))
        else:
            self.conv_layer2 = nn.Conv2d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/2),int(Data_shape1/n_patch/2)],
                                         stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        #out = self.max_pool1(out)
        #print('pool1:',out.shape)
        
        out = self.conv_layer2(out)
        #print('conv2:',out.shape)
        
        #out = self.max_pool2(out)
        #print('pool2:',out.shape)
        
        out = out.permute(0,2,3,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,3,1,2)
        return out   
    

class DeconvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=64,Data_shape1=64,Data_shape2=64,n_patch=8):
        super(DeconvNeuralNet, self).__init__()
        self.deconv_layer1 = nn.ConvTranspose2d(dim, hidden_dim,
                                     kernel_size=[int(Data_shape1/n_patch/2),int(Data_shape1/n_patch/2)],
                                    stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        if int(Data_shape1/n_patch/4) > 1:
            self.deconv_layer2 = nn.ConvTranspose2d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/4),int(Data_shape1/n_patch/4)],
                                         stride=int(Data_shape1/n_patch/4))
        else:
            self.deconv_layer2 = nn.ConvTranspose2d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/2),int(Data_shape1/n_patch/2)],
                                         stride=int(Data_shape1/n_patch/2))
            
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.deconv_layer1(x)
        #print('conv1:',out.shape)
        
        #out = self.max_pool1(out)
        #print('pool1:',out.shape)
        
        out = self.deconv_layer2(out)
        #print('conv2:',out.shape)
        
        #out = self.max_pool2(out)
        #print('pool2:',out.shape)
        
        out = out.permute(0,2,3,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,3,1,2)
        return out  


class ConvNeuralNet1D(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=256,Data_shape1=256,n_patch=32):
        super(ConvNeuralNet1D, self).__init__()
        self.conv_layer1 = nn.Conv1d(dim, hidden_dim,
                                     kernel_size=[int(Data_shape1/n_patch/2)],
                                     stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        if int(Data_shape1/n_patch/4)>1:
            self.conv_layer2 = nn.Conv1d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/4)],
                                         stride=int(Data_shape1/n_patch/4))
        else:
            self.conv_layer2 = nn.Conv1d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/2)],
                                         stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        #out = self.max_pool1(out)
        #print('pool1:',out.shape)
        
        out = self.conv_layer2(out)
        #print('conv2:',out.shape)
        
        #out = self.max_pool2(out)
        #print('pool2:',out.shape)
        
        out = out.permute(0,2,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,2,1)
        return out   
    

class DeconvNeuralNet1D(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=256,Data_shape1=256,n_patch=32):
        super(DeconvNeuralNet1D, self).__init__()
        self.deconv_layer1 = nn.ConvTranspose1d(dim, hidden_dim,
                                     kernel_size=[int(Data_shape1/n_patch/2)],
                                    stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        if int(Data_shape1/n_patch/4)>1:
            self.deconv_layer2 = nn.ConvTranspose1d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/4)],
                                         stride=int(Data_shape1/n_patch/4))
        else:
            self.deconv_layer2 = nn.ConvTranspose1d(hidden_dim, out_dim,
                                         kernel_size=[int(Data_shape1/n_patch/2)],
                                         stride=int(Data_shape1/n_patch/2))
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.deconv_layer1(x)
        #print('conv1:',out.shape)
        
        #out = self.max_pool1(out)
        #print('pool1:',out.shape)
        
        out = self.deconv_layer2(out)
        #print('conv2:',out.shape)
        
        #out = self.max_pool2(out)
        #print('pool2:',out.shape)
        
        out = out.permute(0,2,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,2,1)
        return out 
    
    
# class Decoder_NN(nn.Module):
#     def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
#         super(Decoder_NN, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.n_layers = len(shapes) - 1
#         self.shapes = shapes
#         self.first = nn.Linear(in_dim,shapes[0])
#         self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
#         self.last = nn.Linear(shapes[-1], out_dim)
#         self.NL = NL(inplace=True) 
        
#     def forward(self, y):
#         y_in = y.permute(0,2,1)
#         y = self.NL(self.first.forward(y_in))
#         for layer in self.layers:
#             y = self.NL(layer.forward(y))   
#         y_out = self.last.forward(y)
#         y = y_out.permute(0,2,1)

#         return y


class Decoder_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(Decoder_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y.permute(0,2,1,3)
        y_in = y_in.flatten(2,3)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y_out = self.last.forward(y)
        y = y_out.permute(0,2,1)

        return y
    
class Decoder_NN_2D(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,shapes_out,NL=nn.ELU,):
        super(Decoder_NN_2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.shapes_out = shapes_out
        self.NL = NL(inplace=True)
        # self.dropout = nn.Dropout(dropout) #ADD
        
    def forward(self, y):
        y_in = y.flatten(start_dim=1,end_dim=-1)
        y = self.NL(self.first.forward(y_in))
        # y = self.dropout(y) #Add
        for layer in self.layers:
            y = self.NL(layer.forward(y)) 
            # y = self.dropout(y) #Add
        y = self.last.forward(y)
        y_out = y.reshape(y.shape[0],self.shapes_out[0],self.shapes_out[1],self.shapes_out[2])
        

        return y_out
    
    
class BrainConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 K1 = (16,16,2),
                 K2 = (16,16,2),
                 S1 = (8,7,2),
                 S2 = (3,2,1)):
        super(BrainConvNeuralNet, self).__init__()
        
        self.conv_layer1 = nn.Conv3d(dim, hidden_dim,
                                     kernel_size=K1,
                                     stride=S1
                                    )
        

        self.conv_layer2 = nn.Conv3d(hidden_dim, out_dim,
                                         kernel_size=K2,
                                         stride=S2
                                        )
        
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        x = x.permute(0,4,1,2,3)
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        #out = self.max_pool1(out)
        #print('pool1:',out.shape)
        
        out = self.conv_layer2(out)
        #print('conv2:',out.shape)
        
        #out = self.max_pool2(out)
        #print('pool2:',out.shape)
        
        out = out.permute(0,2,3,4,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out   


class SingleConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=64,K=[4,4],S=[4,4]):
        super(SingleConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(dim, hidden_dim,
                                     kernel_size=K,
                                     stride=S)
        
 
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        
        out = out.permute(0,2,3,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,3,1,2)
        return out   
    
class Single3DConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, out_dim=32,hidden_ff=64,K=[4,4,5],S=[4,4,1]):
        super(SingleConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv3d(dim, hidden_dim,
                                     kernel_size=K,
                                     stride=S)
        
 
        
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        #print('conv1:',out.shape)
        
        
        out = out.permute(0,2,3,4,1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,4,1,2,3)
        return out   

    
class Conv2D_interpolation(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, dim, hidden_dim=32, 
                 out_dim=32,hidden_ff=64,
                 K=[8,8],
                 S=[8,8],
                 total_time=10
                ):
        
        super(Conv2D_interpolation, self).__init__()
        
        self.conv_layer = nn.Conv2d(dim, hidden_dim,
                                     kernel_size=K,
                                    stride=S
                                    )
        
        self.fc1 = nn.Linear(hidden_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
        self.total_time = total_time
       
    def forward(self, x):
        x = x.permute(0,3,4,1,2)
        out = torch.cat([self.conv_layer(x[:,i,...]).unsqueeze(1) for i in range(x.shape[1])],dim=1)
        
        out = out.permute(0,3,4,1,2)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        out = torch.nn.functional.interpolate(
            out.permute(0,4,1,2,3),
            size=[out.shape[1],
                  out.shape[2],
                  self.total_time],
            mode='trilinear')
        out = out.permute(0,2,3,4,1)
        
        return out      

class DualConvConcatNet(nn.Module):
    def __init__(self, dim, hidden_dim=32, out_dim=32, hidden_ff=64,
                 K1=[3, 3], S1=[2, 2],
                 K2=[5, 5], S2=[2, 2]):
        super(DualConvConcatNet, self).__init__()
        
        # First CNN branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=K1, stride=S1, padding=1),
            nn.ReLU()
        )
        
        # Second CNN branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=K2, stride=S2, padding=2),
            nn.ReLU()
        )
        
        # Output MLP after concatenation
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)

    def forward(self, x):
        # Input: x [B, C, H, W]
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        # Match spatial sizes
        if out1.shape[-2:] != out2.shape[-2:]:
            H = min(out1.shape[-2], out2.shape[-2])
            W = min(out1.shape[-1], out2.shape[-1])
            out1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=False)
            out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)

        # Concatenate along channel dim -> [B, 2*hidden_dim, H, W]
        out = torch.cat([out1, out2], dim=1)

        # Move channels to last dim to apply FC
        out = out.permute(0, 2, 3, 1)  # [B, H, W, 2*hidden_dim]
        out = self.fc1(out)            # [B, H, W, hidden_ff]
        out = self.relu(out)
        out = self.fc2(out)            # [B, H, W, out_dim]

        # Move back to [B, C, H, W]
        out = out.permute(0, 3, 1, 2)  # [B, out_dim, H, W]
        return out

        
# class DualConvConcatNet(nn.Module):
#     def __init__(self, dim, hidden_dim=32, out_dim=32, hidden_ff=64,
#                  K1=[3, 3], S1=[2, 2],
#                  K2=[5, 5], S2=[2, 2]):
#         super(DualConvConcatNet, self).__init__()
        
#         # First CNN branch
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, kernel_size=K1, stride=S1, padding=1),
#             nn.ReLU()
#         )
        
#         # Second CNN branch
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, kernel_size=K2, stride=S2, padding=2),
#             nn.ReLU()
#         )
        
#         # Output MLP after concatenation
#         self.fc1 = nn.Linear(2 * hidden_dim, hidden_ff)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_ff, out_dim)

#     def forward(self, x):
#         # Input shape: [B, H, W, 1, C]
#         # if x.ndim == 5:
#         #     x = x.squeeze(3)  # Remove dummy time dim → [B, H, W, C]
#         # assert x.ndim == 4, f"Expected 4D input after squeeze, got {x.shape}"

#         # x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

#         out1 = self.branch1(x)
#         out2 = self.branch2(x)

#         # Align spatial shapes if needed
#         if out1.shape[-2:] != out2.shape[-2:]:
#             H = min(out1.shape[-2], out2.shape[-2])
#             W = min(out1.shape[-1], out2.shape[-1])
#             out1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=False)
#             out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)

#         out = torch.cat([out1, out2], dim=1)           # [B, 2*hidden_dim, H, W]
#         out = out.permute(0, 2, 3, 1)                  # [B, H, W, C]
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = out.unsqueeze(3)                         # [B, H, W, 1, C_out]
#         return out

class EfficientCrackDecoder(nn.Module):
    """
    Memory-efficient decoder with proper shape handling
    """
    def __init__(self, spatial_size=128, in_channels=16, out_channels=2):
        super().__init__()
        
        self.spatial_size = spatial_size
        self.in_channels = in_channels
        
        # Project to higher channels
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Upsample from 128x128 to 256x256
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Refinement at 256x256
        self.refine = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.output = nn.Conv2d(8, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: from IE solver, shape can be:
               [B, H, W, C] or [B, H, W, 1, C] or [B, 1, H, W, C]
        Returns:
            [B, 2, 256, 256]
        """
        # Handle different input shapes from IE solver
        if x.dim() == 5:
            # [B, H, W, 1, C] or [B, 1, H, W, C]
            if x.shape[3] == 1:
                x = x.squeeze(3)  # [B, H, W, C]
            elif x.shape[1] == 1:
                x = x.squeeze(1)  # [B, H, W, C]
        
        # Now x should be [B, H, W, C]
        if x.dim() != 4:
            raise ValueError(f"After squeezing, expected 4D tensor, got shape {x.shape}")
        
        B, H, W, C = x.shape
        
        # Permute to [B, C, H, W] for convolutions
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Project
        x = self.project(x)  # [B, 64, H, W]
        
        # Upsample
        x = self.upsample(x)  # [B, 32, 256, 256]
        
        # Refine
        x = self.refine(x)  # [B, 8, 256, 256]
        
        # Output
        x = self.output(x)  # [B, 2, 256, 256]
        
        return x