import os
import shutil

def save_checkpoint(state, is_best, epoch, checkpoint):
    filename = os.path.join(checkpoint, f'last_{epoch}.pth.tar')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint, f'model_best_{epoch}.pth.tar'))