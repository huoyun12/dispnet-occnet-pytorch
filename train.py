import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from models.dispnet import DispNet
from models.occnet import OccNet
from utils.losses import FullLoss
from data.lightfield_dataset import get_dataloader


def warp_image(image, disparity):
    B, C, H, W = image.shape
    device = image.device
    
    print(f"[DEBUG warp_image (train.py)] image: {image.shape}, disparity: {disparity.shape}")

    if disparity.dim() == 3:
        disparity = disparity.unsqueeze(1)
    
    if disparity.shape[2] != H or disparity.shape[3] != W:
        print(f"[DEBUG warp_image] Interpolating disparity from {disparity.shape[2:]} to {(H, W)}")
        disparity = torch.nn.functional.interpolate(disparity, size=(H, W), mode='bilinear', align_corners=False)

    print(f"[DEBUG warp_image] After interpolate, disparity: {disparity.shape}")

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device),
                                     torch.arange(H, device=device), indexing='xy')

    grid_x = grid_x.float().unsqueeze(2).unsqueeze(0).repeat(B, 1, 1, 1)
    grid_y = grid_y.float().unsqueeze(2).unsqueeze(0).repeat(B, 1, 1, 1)

    print(f"[DEBUG warp_image] grid_x: {grid_x.shape}, grid_y: {grid_y.shape}")

    disp = disparity[:, 0:1, :, :].permute(0, 2, 3, 1)
    
    print(f"[DEBUG warp_image] disp: {disp.shape}")

    new_x = grid_x + disp
    
    print(f"[DEBUG warp_image] new_x: {new_x.shape}")

    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * grid_y / (H - 1) - 1.0

    grid = torch.cat([new_x, new_y], dim=3)
    
    print(f"[DEBUG warp_image] grid: {grid.shape}")

    warped = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    print(f"[DEBUG warp_image] warped: {warped.shape}")

    return warped


def train_epoch(dispnet, occnet, criterion, dataloader, optimizer, device, epoch):
    dispnet.train()
    occnet.train()

    total_loss = 0.0
    loss_stats = {'wpm': 0.0, 'rec': 0.0, 'ssim': 0.0, 'smd': 0.0, 'smo': 0.0, 'total': 0.0}

    for batch_idx, batch in enumerate(dataloader):
        img_center = batch['img_center'].to(device)
        img_left = batch['img_left'].to(device)
        img_right = batch['img_right'].to(device)

        optimizer.zero_grad()

        coarse_disparity, refined_disparity, features = dispnet(img_center, img_left, img_right)
        
        print(f"[DEBUG train_epoch] coarse_disparity: {coarse_disparity.shape}, refined_disparity: {refined_disparity.shape}, features: {features.shape}")

        warped_left = warp_image(img_left, refined_disparity)
        warped_right = warp_image(img_right, -refined_disparity)
        
        print(f"[DEBUG] warped_left: {warped_left.shape}, warped_right: {warped_right.shape}")

        if refined_disparity.dim() == 3:
            refined_disparity = refined_disparity.unsqueeze(1)
        if refined_disparity.shape[2] != warped_left.shape[2]:
            refined_disparity = torch.nn.functional.interpolate(refined_disparity, size=warped_left.shape[2:], mode='bilinear', align_corners=False)
        
        print(f"[DEBUG] refined_disparity for cat: {refined_disparity.shape}")

        occ_input = torch.cat([warped_left, warped_right, refined_disparity], dim=1)
        
        print(f"[DEBUG] occ_input: {occ_input.shape}")

        occ_left, occ_right = occnet(occ_input)

        loss, loss_dict = criterion(img_center, img_left, img_right,
                                   coarse_disparity, refined_disparity,
                                   occ_left, occ_right)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_stats[k] += v

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    for k in loss_stats:
        loss_stats[k] /= len(dataloader)

    return avg_loss, loss_stats


def validate(dispnet, occnet, criterion, dataloader, device):
    dispnet.eval()
    occnet.eval()

    total_loss = 0.0
    loss_stats = {'wpm': 0.0, 'rec': 0.0, 'ssim': 0.0, 'smd': 0.0, 'smo': 0.0, 'total': 0.0}

    with torch.no_grad():
        for batch in dataloader:
            img_center = batch['img_center'].to(device)
            img_left = batch['img_left'].to(device)
            img_right = batch['img_right'].to(device)

            coarse_disparity, refined_disparity, features = dispnet(img_center, img_left, img_right)

            warped_left = warp_image(img_left, refined_disparity)
            warped_right = warp_image(img_right, -refined_disparity)

            occ_input = torch.cat([warped_left, warped_right, refined_disparity], dim=1)

            occ_left, occ_right = occnet(occ_input)

            loss, loss_dict = criterion(img_center, img_left, img_right,
                                       coarse_disparity, refined_disparity,
                                       occ_left, occ_right)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_stats[k] += v

    avg_loss = total_loss / len(dataloader)
    for k in loss_stats:
        loss_stats[k] /= len(dataloader)

    return avg_loss, loss_stats


def main():
    parser = argparse.ArgumentParser(description='Train DispNet and OccNet for Light Field Depth Estimation')
    parser.add_argument('--data_dir', type=str, default='./data/hci', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default=None, help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (minimum 2 for BatchNorm)')
    parser.add_argument('--crop_size', type=int, default=256, help='Crop size for training')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--max_channels', type=int, default=128, help='Maximum number of channels')
    parser.add_argument('--num_disparities', type=int, default=25, help='Number of disparity samples')
    parser.add_argument('--num_residual_disparities', type=int, default=21, help='Number of residual disparity samples')
    parser.add_argument('--angular_size', type=int, default=7, help='Angular size of light field')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if HAS_TENSORBOARD:
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None
        print("Warning: tensorboard not available, logging disabled")

    print("Initializing models...")
    dispnet = DispNet(
        in_channels=3,
        max_channels=args.max_channels,
        num_disparities=args.num_disparities,
        num_residual_disparities=args.num_residual_disparities,
        num_cost_filters=2
    ).to(device)

    occnet = OccNet(
        in_channels=7,
        max_channels=64
    ).to(device)

    dispnet_params = sum(p.numel() for p in dispnet.parameters())
    occnet_params = sum(p.numel() for p in occnet.parameters())
    print(f"DispNet parameters: {dispnet_params / 1e6:.3f}M")
    print(f"OccNet parameters: {occnet_params / 1e6:.3f}M")
    print(f"Total parameters: {(dispnet_params + occnet_params) / 1e6:.3f}M")

    criterion = FullLoss(alpha_ssim=1.0, alpha_smd=0.1, alpha_smo=0.05, eta=100.0)

    optimizer = optim.Adam(
        list(dispnet.parameters()) + list(occnet.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            dispnet.load_state_dict(checkpoint['dispnet_state_dict'])
            occnet.load_state_dict(checkpoint['occnet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")

    print("Loading datasets...")
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        split='train',
        crop_size=args.crop_size,
        angular_size=args.angular_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = None
    if args.val_dir:
        val_loader = get_dataloader(
            data_dir=args.val_dir,
            batch_size=args.batch_size,
            split='val',
            crop_size=args.crop_size,
            angular_size=args.angular_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    print(f"Training samples: {len(train_loader.dataset)}")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_stats = train_epoch(dispnet, occnet, criterion, train_loader, optimizer, device, epoch + 1)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"  WPM: {train_stats['wpm']:.4f}, REC: {train_stats['rec']:.4f}, SSIM: {train_stats['ssim']:.4f}")
        print(f"  SMD: {train_stats['smd']:.4f}, SMO: {train_stats['smo']:.4f}")

        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)

        if val_loader:
            val_loss, val_stats = validate(dispnet, occnet, criterion, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
            if writer is not None:
                writer.add_scalar('Loss/val', val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'dispnet_state_dict': dispnet.state_dict(),
                    'occnet_state_dict': occnet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

        scheduler.step()

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'dispnet_state_dict': dispnet.state_dict(),
                'occnet_state_dict': occnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    if writer is not None:
        writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
