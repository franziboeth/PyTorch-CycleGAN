import argparse
import itertools
import mlflow
import mlflow.pytorch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import log_image
from utils import weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
train_transforms = transforms.Compose([transforms.Resize((int(opt.size), int(opt.size)), Image.BICUBIC),  
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])

val_test_transforms = transforms.Compose([transforms.Resize((int(opt.size), int(opt.size)), Image.BICUBIC), 
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])

train_dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=train_transforms, unaligned=True, mode="train"), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=0)

val_dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=val_test_transforms, unaligned=True, mode="val"), 
                        batch_size=1, shuffle=True, num_workers=opt.n_cpu)

test_dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=val_test_transforms, unaligned=True, mode="test"), 
                        batch_size=1, shuffle=True, num_workers=opt.n_cpu)


mlflow.set_tracking_uri("file:///mnt/lustre/work/ritter/rvy682/mlruns")
mlflow.set_experiment("cycleGAN")
mlflow.start_run(run_name=f"run_{opt.epoch}_to_{opt.n_epochs}")
mlflow.log_params({
        "n_epochs": opt.n_epochs,
        "batch_size": opt.batchSize,
        "learning_rate": opt.lr,
        "decay_epoch": opt.decay_epoch,
        "image_size": opt.size,
        "input_nc": opt.input_nc,
        "output_nc": opt.output_nc
    })

###################################

###### Training Loop ######
for epoch in range(opt.epoch, opt.n_epochs):
    netG_A2B.train()
    netG_B2A.train()

    epoch_loss_G = 0.0
    epoch_loss_G_identity = 0.0
    epoch_loss_G_GAN = 0.0
    epoch_loss_G_cycle = 0.0
    epoch_loss_D = 0.0
    numberbatches_train = 0

    for i, batch in enumerate(train_dataloader):
        # Set model input
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*0.5
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*0.5

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        
        ################################

        epoch_loss_G += loss_G.item()
        epoch_loss_G_identity += (loss_identity_A + loss_identity_B).item()
        epoch_loss_G_GAN += (loss_GAN_A2B + loss_GAN_B2A).item()
        epoch_loss_G_cycle += (loss_cycle_ABA + loss_cycle_BAB).item()  
        epoch_loss_D += (loss_D_A + loss_D_B).item()
        numberbatches_train +=1 


        print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(train_dataloader)}] "
              f"loss_G={epoch_loss_G:.4f} loss_D={epoch_loss_D:.4f}")

        if epoch % 5 == 0 and i == 0:
            log_image(real_A, f"epoch_{epoch}_real_A.png")
            log_image(real_B, f"epoch_{epoch}_real_B.png")
            log_image(fake_A, f"epoch_{epoch}_fake_A.png")
            log_image(fake_B, f"epoch_{epoch}_fake_B.png")

    if numberbatches_train > 0:
        avg_train_metrics = {
            "train/loss_G": epoch_loss_G/ numberbatches_train,
            "train/loss_G_identiy": epoch_loss_G_identity / numberbatches_train,
            "train/loss_G_GAN": epoch_loss_G_GAN / numberbatches_train,
            "train/loss_G_cycle": epoch_loss_G_cycle / numberbatches_train,
            "train/loss_D": epoch_loss_D / numberbatches_train
        }
        mlflow.log_metrics(avg_train_metrics, step=epoch)
    
    # Save models checkpoints
    # mlflow.pytorch.log_model(netG_A2B, "netG_A2B")
    # mlflow.pytorch.log_model(netG_B2A, "netG_B2A")
    # mlflow.pytorch.log_model(netD_A, "netD_A")
    # mlflow.pytorch.log_model(netD_B, "netD_B")

    ###################################

    ###### Validation Loop #####
    netG_A2B.eval()
    netG_B2A.eval()

    loss_val = 0.0
    numberbatches_val = 0

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            real_A = Variable(batch['A'].type(Tensor)) 
            real_B = Variable(batch['B'].type(Tensor))

            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)

            recovered_A = netG_B2A(fake_B)
            recovered_B = netG_A2B(fake_A)

            loss_cycle_val = (criterion_cycle(recovered_A, real_A) + criterion_cycle(recovered_B, real_B)) * 10.0
            loss_identity_val = criterion_identity(netG_A2B(real_B), real_B) * 0.5 + criterion_identity(netG_B2A(real_A), real_A) * 0.5
            val_loss_batch = loss_cycle_val + loss_identity_val

            loss_val += val_loss_batch.item()
            numberbatches_val += 1
            
            # if i == 0:
            #     log_image(real_A, f"val_epoch_{epoch}_real_A.png")
            #     log_image(real_B, f"val_epoch_{epoch}_real_B.png")
            #     log_image(fake_A, f"val_epoch_{epoch}_fake_A.png")
            #     log_image(fake_B, f"val_epoch_{epoch}_fake_B.png")
        
        if numberbatches_val > 0:
            avg_val_metrics = {
                "val/loss_val": loss_val / numberbatches_val
            }
            mlflow.log_metrics(avg_val_metrics, step=epoch)

        
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    torch.save(netG_A2B.state_dict(), f"/mnt/lustre/work/ritter/rvy682/outputs/netG_A2B.pt")
    torch.save(netG_B2A.state_dict(), f"/mnt/lustre/work/ritter/rvy682/outputs/netG_B2A.pt")
    torch.save(netD_A.state_dict(), f"/mnt/lustre/work/ritter/rvy682/outputs/netD_A.pt")
    torch.save(netD_B.state_dict(), f"/mnt/lustre/work/ritter/rvy682/outputs/netD_B.pt")
    
##### Testing Loop ######
netG_A2B.eval()
netG_B2A.eval()

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        fake_B = netG_A2B(real_A)
        fake_A = netG_B2A(real_B)

        log_image(fake_B, f"test_fakeB_{i}.png")
        log_image(fake_A, f"test_fakeA_{i}.png")
        
mlflow.end_run()

        






        