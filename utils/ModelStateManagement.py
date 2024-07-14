import torch


def save_checkpoint(model, save_file="./checkpoint.pth"):
    torch.save(model.state_dict(), save_file)


def load_checkpoint(model, save_file="./checkpoint.pth"):
    dictionary = torch.load(save_file)
    model.load_state_dict(dictionary)
    return model
