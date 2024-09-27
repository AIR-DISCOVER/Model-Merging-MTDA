import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

a = torch.rand(50000, 50000).to(device)

for i in range(10000):
    a = a * a
    
    if i % 1000 == 0:
        print(i, a)

print(a)
