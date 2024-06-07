import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.utils as vutils




# Set random seed for reproducibility
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
# Parameters to define the model.


class GANConfig:
    batch_size = 128
    image_size = 64  # Spatial size of training images.
    nc = 3           # Number of channels in the training images.
    nz = 100         # Size of z latent vector (i.e., size of generator input).
    ngf = 64         # Size of feature maps in generator.
    ndf = 64         # Size of feature maps in discriminator.
    num_epochs = 5   # Number of training epochs.
    lr = 0.0002      # Learning rate for optimizers.
    beta1 = 0.5      # Beta1 hyperparam for Adam optimizers.
    ngpu = 1         # Number of GPUs available. Use 0 for CPU mode.

config = GANConfig()

# Load dataset
dataroot = r'C:\Users\aayus\PycharmProjects\Discriminator_comparison\dataset\img_align_celeba'
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(config.image_size),
                               transforms.CenterCrop(config.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

# Initialize device
device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")

# Define the Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(config.nz, config.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(config.ngf * 8, config.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(config.ngf * 4, config.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(config.ngf * 2, config.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(config.ngf, config.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(config.nc, config.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(config.ndf, config.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(config.ndf * 2, config.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(config.ndf * 4, config.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(config.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def inception_score(preds, num_splits=10):
    scores = []
    for index in range(num_splits):
        part = preds[index * (len(preds) // num_splits):(index + 1) * (len(preds) // num_splits)]
        kl = part * (torch.log(part) - torch.log(part.mean(dim=0, keepdim=True)))
        scores.append(kl.sum(dim=1).mean().exp())
    is_mean, is_std = torch.mean(torch.stack(scores)), torch.std(torch.stack(scores))
    return is_mean.item(), is_std.item()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    # Create the generator and discriminator
    nz = 100  # Ensure this matches the expected dimension for noise input in G
    netG = Generator(config.ngpu).to(device)
    netD = Discriminator(config.ngpu).to(device)

    # Define test noise
    test_noise = torch.randn(64, nz, 1, 1, device=device)  # Adjust the batch size for generated images if needed

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (config.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(config.ngpu)))
        netD = nn.DataParallel(netD, list(range(config.ngpu)))

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # Training Loop
    print("Starting Training Loop...")
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    for epoch in range(config.num_epochs):
        for i, data in enumerate(dataloader, 0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ## Train with all-real batch
            real_images = data[0].to(device)
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1.0, device=device)

            netD.zero_grad()
            output = netD(real_cpu).view(-1)
            errD_real = nn.BCELoss()(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = nn.BCELoss()(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(1)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = nn.BCELoss()(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z1)): %.4f / D(G(z2)): %.4f'
                      % (epoch, config.num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # Generate fake images periodically to monitor progress
        if epoch % 1 == 0:  # Adjust the frequency as needed
            with torch.no_grad():
                fake_images = netG(test_noise).detach()
                img_list.append(fake_images)

    # Display or save training progress images if necessary here

        # Calculate FID and IS
        if i % 100 == 0:  # Adjust this to run less often
            with torch.no_grad():
                fake_images = netG(test_noise).detach()
                real_features = model(real_images)
                fake_features = (fake_images)
                fid_value = calculate_fid(real_features.cpu().numpy(), fake_features.cpu().numpy())
                is_value, is_std = inception_score(fake_features.softmax(dim=1))
                print(
                    f'Epoch [{epoch}/{config.num_epochs}] Batch {i}/{len(dataloader)} FID: {fid_value:.2f}, IS: {is_value:.2f} ± {is_std:.2f}')

    print("Training finished.")

def display_images(fake_images):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(fake_images.to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.show()

def display_images_over_time(img_list, G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Show images as they evolved through training
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(img_list[0], padding=5, normalize=True).cpu(), (1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images After Training")
    plt.imshow(np.transpose(vutils.make_grid(img_list[-1], padding=5, normalize=True), (1,2,0)))
    plt.show()

if __name__ == '__main__':
    main()