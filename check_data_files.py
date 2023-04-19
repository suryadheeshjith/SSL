import os.path as osp

try:
    assert True == osp.isdir("dataset/dataset/train")
    assert True == osp.isdir("dataset/dataset/unlabeled")
    assert True == osp.isdir("dataset/dataset/val")

except:
    print("Error in accessing datasets")
    exit(0)

print("All datasets present!")