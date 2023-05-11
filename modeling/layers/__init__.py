import torch
from modeling.layers.deviation_loss import DeviationLoss, DeviationLossCpu
from modeling.layers.binary_focal_loss import BinaryFocalLoss

def build_criterion(criterion):
    if criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss()
    elif criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "focal":
        print("Loss : Focal")
        return BinaryFocalLoss()
    else:
        raise NotImplementedError
def build_criterion_cpu(criterion):
    if criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLossCpu()
    elif criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "focal":
        print("Loss : Focal")
        return BinaryFocalLoss()
    else:
        raise NotImplementedError