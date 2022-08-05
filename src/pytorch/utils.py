import os
import torch
from collections import OrderedDict
from src.pytorch.config import ARTIFACTS_DIR, PYTORCH_FILE_NAME, ONNX_FILE_NAME


# onnx export

# # Input to the model
# x = torch.randn(1, 3, 224, 224, requires_grad=True)

# # Export the model
# torch.onnx.export(onnx_model,               # model being run
#                   # model input (or a tuple for multiple inputs)
#                   x,
#                   # where to save the model (can be a file or file-like object)
#                   "om-detection.onnx",
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   )


def save_checkpoint(state: OrderedDict, filename=PYTORCH_FILE_NAME):
    """This function saves pytorch model as a serialized object to disk

    Args:
        state (OrderedDict): model state
        filename (str, optional): Model checkpoint file name. Defaults to PYTORCH_FILE_NAME.
    """
    print("=> Saving checkpoint..")
    dest_path = os.path.join(ARTIFACTS_DIR, PYTORCH_FILE_NAME)
    torch.save(state, dest_path)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
