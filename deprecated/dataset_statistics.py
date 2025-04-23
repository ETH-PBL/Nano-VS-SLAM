
from datasets.scene_parse_150 import get_dataset
import torch
def count_classes_fast(tensor, num_classes=151):
    """
    Count the occurrences of each class (0 to num_classes-1) in a PyTorch tensor
    using fast matrix operations.

    Args:
    tensor (torch.Tensor): The input tensor.
    num_classes (int): The number of classes to count.

    Returns:
    torch.Tensor: A tensor of shape (num_classes,) where each element represents
                  the count of a class in the input tensor.
    """
    # Flatten the tensor to a 1D array for processing
    flattened = tensor.flatten()

    # Use bincount which is much faster for counting occurrences
    counts = torch.bincount(flattened, minlength=num_classes)

    # Trim counts to num_classes if needed (bincount might return a longer vector)
    return counts[:num_classes]

dataset = get_dataset()
i = 0
t = torch.zeros(151).double()
for sample in dataset:
    h = count_classes_fast(sample["seg"].int()).float()
    h = h/h.sum()

    t += h
    i+=1
    print(i,t)
print(t/i)

class_frequency = [0.0904, 0.1495, 0.1002, 0.0941, 0.0605, 0.0509, 0.0417, 0.0358, 0.0198,
        0.0186, 0.0180, 0.0160, 0.0132, 0.0161, 0.0149, 0.0105, 0.0106, 0.0164,
        0.0108, 0.0093, 0.0096, 0.0083, 0.0071, 0.0064, 0.0053, 0.0069, 0.0067,
        0.0085, 0.0044, 0.0039, 0.0069, 0.0039, 0.0048, 0.0031, 0.0034, 0.0039,
        0.0030, 0.0022, 0.0020, 0.0024, 0.0019, 0.0025, 0.0022, 0.0022, 0.0018,
        0.0017, 0.0020, 0.0025, 0.0016, 0.0046, 0.0014, 0.0015, 0.0017, 0.0018,
        0.0015, 0.0011, 0.0020, 0.0016, 0.0013, 0.0011, 0.0018, 0.0028, 0.0014,
        0.0019, 0.0009, 0.0012, 0.0011, 0.0015, 0.0015, 0.0016, 0.0015, 0.0010,
        0.0011, 0.0012, 0.0009, 0.0009, 0.0011, 0.0011, 0.0012, 0.0009, 0.0008,
        0.0007, 0.0007, 0.0008, 0.0008, 0.0007, 0.0007, 0.0008, 0.0006, 0.0008,
        0.0006, 0.0006, 0.0007, 0.0008, 0.0005, 0.0009, 0.0007, 0.0005, 0.0005,
        0.0007, 0.0004, 0.0006, 0.0007, 0.0006, 0.0005, 0.0005, 0.0005, 0.0006,
        0.0006, 0.0006, 0.0008, 0.0005, 0.0005, 0.0005, 0.0009, 0.0006, 0.0004,
        0.0003, 0.0006, 0.0004, 0.0004, 0.0005, 0.0004, 0.0004, 0.0004, 0.0003,
        0.0004, 0.0004, 0.0003, 0.0006, 0.0004, 0.0004, 0.0003, 0.0004, 0.0003,
        0.0003, 0.0003, 0.0002, 0.0004, 0.0002, 0.0002, 0.0003, 0.0003, 0.0003,
        0.0003, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002]