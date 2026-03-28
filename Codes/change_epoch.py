import torch

ckpt = torch.load("../Models/fno_checkpoint.pth", weights_only=False)

print("Current epoch:", ckpt["epoch"])  # verify before changing

ckpt["epoch"] = 1000  # set whatever epoch you want

torch.save(ckpt, "../Models/fno_checkpoint.pth")
print("Done. Epoch set to", ckpt["epoch"])