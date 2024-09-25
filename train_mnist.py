import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus.grad_sample.grad_sample_module import GradSampleModule
from torchvision import datasets, transforms
from tqdm import tqdm
from Compress_DJW import *
import itertools
import os
import warnings
warnings.filterwarnings("ignore")

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = nn.ZeroPad2d(2)(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output
    
# Convert gradients into vectors
def grad_to_vec(model):
    res = []
    for p in model.parameters():
        res.append(p.grad_sample.view(p.grad_sample.size(0), -1))
    return torch.cat(res, dim=1).squeeze()

# Clipt L2 norm of gradient to be at most C
def clip(grad, C):
    vec_norm = grad.norm(2, 1)
    multiplier = vec_norm.new(vec_norm.size()).fill_(1)
    multiplier[vec_norm.gt(C)] = C / vec_norm[vec_norm.gt(C)]
    grad *= multiplier.unsqueeze(1)
    return grad


# Privatizes the input gradients according to the privacy parameters
def privatize_grad_gpu(args, grad, device, mechanism=None):
    C = args.clip_val
    batch_size = grad.size(0)
    d = grad.size(1)
    grad_vec = torch.zeros((grad.size(0),d+1)).to(device)
    grad_vec[:,:-1] = grad.detach()
    # Complete norm to C then normalize
    v_m = torch.min(C**2 - torch.sum(grad_vec**2,1)+0.0001).to(device)
    if v_m > 0:
        grad_vec[:,-1] = torch.sqrt(C**2 - torch.sum(grad_vec**2,1)+0.0001).to(device)
    grad_vec = grad_vec/C
    epsilon = args.epsilon   

    if mechanism == 'Gaussian' or mechanism == 'nonPrivate':
        grad_vec = None
        grad += torch.randn_like(grad).to(device) * args.clip_val * args.sigma
    else:
        W = None
        for i in range(grad_vec.shape[0]):
            k = 1
            grad_vec[i] = privatize_vector(grad_vec[i],epsilon,k,mechanism,device, args.p,args.gamma,args.sigma,True,W,use_GPU = True)
    if grad_vec is not None:
        grad = grad_vec[:,:-1].to(device).float()
    return C*grad.mean(0)

# updates the gradient of the model to be equal to the input gradient
def update_grad(model, grad):
    model.zero_grad()
    for p in model.parameters():
        size = p.data.view(1,-1).size(1)
        p.grad = grad[:size].view_as(p.data).clone()
        grad = grad[size:]
    return

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    epsilon = args.epsilon
    ran_batch_idx = np.random.randint(len(train_loader))
    
    # Local computation process
    data, target = next(itertools.islice(train_loader, ran_batch_idx, ran_batch_idx + 1))
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    grad = grad_to_vec(model)

    if args.mechanism != 'nonPrivate':
        grad = clip(grad, args.clip_val)

    grad_priv = privatize_grad_gpu(args, grad, device, args.mechanism)
    update_grad(model, grad_priv)

    optimizer.step()
    losses.append(loss.item())
    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(local ε = {epsilon:.2f})"
    )


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in (test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="MNIST Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=3200,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="Momentum",
    )
    parser.add_argument(
        "-c",
        "--clip-val",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this l2 norm",
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default='CompressLDP',
        help="Privacy mechanism",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        metavar="S",
        help="LDP budget",
    )
    parser.add_argument(
        "--num-rep",
        type=int,
        default=1,
        help="Number of Repetitions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--dir-res",
        type=str,
        default="MNIST_results",
        help="Directory for saving results",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        help="model: CNN for MNIST",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    print(torch.cuda.is_available())

    args.p = None
    args.sigma = None
    args.gamma = None
    
    print("Mechanism is " + str(args.mechanism))
    
    model_str = "Mnist_CNN"

    output_file = "%s/mech_%s_num_rep_%d_epochs_%d_lr_%.2e_clip_%.2e_epsilon_%.2e_batch_%d_model_%s.pth" % (
        args.dir_res, args.mechanism, args.num_rep, args.epochs, args.lr, args.clip_val,
        args.epsilon, args.batch_size, model_str
    )
    
    if os.path.exists(output_file):
        print('Existing result')
        return
        
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = np.zeros((args.epochs,args.num_rep))
    
    for it in range(args.num_rep):
        model = GradSampleModule(Mnist_CNN().to(device))

        num_parameters = sum([np.prod(layer.size()) for layer in model.parameters()])
        print("Num of parameters in model = %d" % num_parameters)
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    

        for epoch in tqdm(range(1, args.epochs + 1)):
            
            train(args, model, device, train_loader, optimizer, epoch)
            test_acc = test(model, device, test_loader) 
            run_results[epoch-1][it] = test_acc
    
           
        np.save(output_file,run_results) 

if __name__ == "__main__":
   main()
    
    
    
