# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from skimage.transform import resize
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
cudnn.benchmark = True


# %%
class CustomImageDataset(Dataset):
    def __init__(self, true_data, generated_data):
        self.true_data = true_data
        self.generated_data = generated_data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.label_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.true_data) + len(self.generated_data)

    def __getitem__(self, idx):
        if idx >= len(self.true_data):
            image = resize(self.generated_data[idx - len(self.true_data)], (224, 224))
            image = self.transform(image)
            label = 0
        else:
            image = resize(self.true_data[idx], (224, 224))
            image = self.transform(image)
            label = 1

        return image, label

# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

train_true = np.load("./u_training_data.npy")[:3000].reshape(3000, 128, 128, 1).repeat(3, -1)[:, 10:-10, 10:-10, :]
test_true = np.load("./u_testing_data.npy")[:1000].reshape(1000, 128, 128, 1).repeat(3, -1)[:, 10:-10, 10:-10, :]
generated_data = np.load("/mnt/data_jenner/tanish/diffusers/examples/unconditional_image_generation/u_data.npy").reshape(4000, 128, 128, 1).repeat(3, -1)[:, 10:-10, 10:-10, :]

train_gen, test_gen = train_test_split(generated_data)
dataloaders = {
    "train": DataLoader(CustomImageDataset(train_true, train_gen), batch_size=32, shuffle=True),
    "val": DataLoader(CustomImageDataset(test_true, test_gen), batch_size=32, shuffle=True),
}
# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
class_names = ["true", "generated"]
dataset_sizes = {"train": 6000, "val": 2000}
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
print(classes)
imshow(out, title="")
# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model
# %%
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# %%
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# %%
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=7)

# %%
visualize_model(model_ft)

# %%
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib as mpl
# %%
target_layers = [model_ft.layer4[-1]]
cam = GradCAMPlusPlus(model=model_ft, target_layers=target_layers)


inputs, classes = next(next(iter(dataloaders['train'])))
out = model_ft(inputs.to(device, dtype=torch.float))
_, predictions = torch.max(out, 1)

input_tensor = (inputs + 1) / 2
input_tensor = input_tensor.permute(0, 2, 3, 1).numpy().astype(np.float32)

grayscale_cam = cam(input_tensor=inputs.to(device, dtype=torch.float), eigen_smooth=True, aug_smooth=True)

for i in range(len(inputs)):
# for i in range(2):
    visualization = show_cam_on_image(input_tensor[i], grayscale_cam[i:i+1].transpose(1,2,0), use_rgb=True)
    plt.subplot(1, 2, 1)
    plt.imshow(visualization)
    plt.title(f"{class_names[1 - classes[i].item()]}: gradcam")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(input_tensor[i][:, :, 1], cmap=mpl.colormaps["magma"])
    plt.title(f"{class_names[1 - classes[i].item()]}: original")
    plt.axis("off")
    plt.savefig(f"./figures/{i}.png")

# %%
input_tensor[i].shape
# %%
# %%
