import os.path as osp

try:
    assert True == osp.isdir("/dataset/dataset/train")
    assert True == osp.isdir("/dataset/dataset/unlabeled")
    assert True == osp.isdir("/dataset/dataset/val")

except:
    print("Error in accessing datasets")
    exit(0)

print("All datasets present!")

try:
    import torch

    print(torch.__file__)
    print(torch.__version__)

    # Is PyTorch using a GPU?
    print("GPU available: ", torch.cuda.is_available())

    # Get the name of the current GPU
    print("Name of current GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))

    # How many GPUs are there?
    print("Number of GPUs: ", torch.cuda.device_count())
except:
    print("Error in importing torch")
    exit(0)