import os
import torch


def save_checkpoint(model, opt, steps, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = save_path + "/" + str(steps) + ".pt"
    torch.save({'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.optimizer.state_dict()}, save_file)


def load_checkpoint(file_path, model, opt):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def load_latest(save_path, model, model_opt):
    if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
        last = sorted(os.listdir(save_path), reverse=True, key=lambda x: int(x.partition(".")[0]))[0]
        last_file = os.path.join(save_path, last)	
        steps = int(last.partition(".")[0])
        load_checkpoint(last_file, model, model_opt)
        print("Loaded from step", steps)
        return steps
    else:
        return 0
