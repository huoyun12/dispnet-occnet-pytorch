import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(window_size, sigma):
    gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def warp_image(image, disparity):
    B, C, H, W = image.shape
    device = image.device

    print(f"[DEBUG] Input - image: {image.shape}, disparity: {disparity.shape}, dim: {disparity.dim()}")

    if disparity.dim() == 2:
        disparity = disparity.unsqueeze(0).unsqueeze(0)
        print(f"[DEBUG] After unsqueeze (2D->4D): {disparity.shape}")
    elif disparity.dim() == 3:
        disparity = disparity.unsqueeze(1)
        print(f"[DEBUG] After unsqueeze (3D->4D): {disparity.shape}")

    if disparity.shape[2] != H or disparity.shape[3] != W:
        print(f"[DEBUG] Interpolating: {disparity.shape[2:]} -> {(H, W)}")
        disparity = F.interpolate(disparity, size=(H, W), mode='bilinear', align_corners=False)

    print(f"[DEBUG] After interpolate: {disparity.shape}")

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device),
                                     torch.arange(H, device=device), indexing='xy')

    print(f"[DEBUG] grid_x shape before unsqueeze: {grid_x.shape}")

    grid_x = grid_x.float().unsqueeze(2).unsqueeze(0).repeat(B, 1, 1, 1)
    grid_y = grid_y.float().unsqueeze(2).unsqueeze(0).repeat(B, 1, 1, 1)

    print(f"[DEBUG] grid_x after unsqueeze+repeat: {grid_x.shape}, grid_y: {grid_y.shape}")

    disp = disparity[:, 0:1, :, :].permute(0, 2, 3, 1)
    print(f"[DEBUG] disp shape after permute: {disp.shape}")

    new_x = grid_x + disp
    print(f"[DEBUG] new_x shape: {new_x.shape}")

    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * grid_y / (H - 1) - 1.0

    print(f"[DEBUG] After normalization - new_x: {new_x.shape}, new_y: {new_y.shape}")

    grid = torch.cat([new_x, new_y], dim=3)
    print(f"[DEBUG] grid shape after cat (dim=3): {grid.shape}")

    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    print(f"[DEBUG] warped output: {warped.shape}")

    return warped


class WeightedPhotometricLoss(nn.Module):
    def __init__(self):
        super(WeightedPhotometricLoss, self).__init__()

    def forward(self, img_left, img_right, img_center, occ_left, occ_right):
        warped_left = warp_image(img_left, img_center)
        warped_right = warp_image(img_right, -img_center)

        error_left = torch.abs(warped_left - img_center)
        error_right = torch.abs(warped_right - img_center)

        loss_left = (occ_left * error_left).mean()
        loss_right = (occ_right * error_right).mean()

        return (loss_left + loss_right) / 2


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, img_center, warped_left, warped_right, occ_left, occ_right):
        recon = occ_left * warped_left + occ_right * warped_right
        loss = F.l1_loss(recon, img_center)
        return loss


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2)


class SmoothnessLoss(nn.Module):
    def __init__(self, eta=100.0):
        super(SmoothnessLoss, self).__init__()
        self.eta = eta

    def forward(self, disparity, img_center):
        if disparity.dim() == 3:
            disparity = disparity.unsqueeze(1)

        grad_disp_x = torch.abs(disparity[:, :, :, :-1] - disparity[:, :, :, 1:])
        grad_disp_y = torch.abs(disparity[:, :, :-1, :] - disparity[:, :, 1:, :])

        grad_img_x = torch.abs(img_center[:, :, :, :-1] - img_center[:, :, :, 1:])
        grad_img_y = torch.abs(img_center[:, :, :-1, :] - img_center[:, :, 1:, :])

        weight_x = torch.exp(-self.eta * grad_img_x)
        weight_y = torch.exp(-self.eta * grad_img_y)

        loss_x = (grad_disp_x * weight_x).mean()
        loss_y = (grad_disp_y * weight_y).mean()

        return loss_x + loss_y


class OcclusionSmoothnessLoss(nn.Module):
    def __init__(self, eta=100.0):
        super(OcclusionSmoothnessLoss, self).__init__()
        self.eta = eta

    def forward(self, occ_left, img_center):
        print(f"[OcclusionSmoothnessLoss DEBUG] occ_left: {occ_left.shape}, img_center: {img_center.shape}")

        if occ_left.shape[2] != img_center.shape[2] or occ_left.shape[3] != img_center.shape[3]:
            print(f"[OcclusionSmoothnessLoss DEBUG] Interpolating occ_left from {occ_left.shape[2:]} to {img_center.shape[2:]}")
            occ_left = F.interpolate(occ_left, size=img_center.shape[2:], mode='bilinear', align_corners=False)

        grad_occ_x = torch.abs(occ_left[:, :, :, :-1] - occ_left[:, :, :, 1:])
        grad_occ_y = torch.abs(occ_left[:, :, :-1, :] - occ_left[:, :, 1:, :])

        grad_img_x = torch.abs(img_center[:, :, :, :-1] - img_center[:, :, :, 1:])
        grad_img_y = torch.abs(img_center[:, :, :-1, :] - img_center[:, :, 1:, :])

        print(f"[OcclusionSmoothnessLoss DEBUG] grad_occ_x: {grad_occ_x.shape}, grad_img_x: {grad_img_x.shape}")

        weight_x = torch.exp(-self.eta * grad_img_x)
        weight_y = torch.exp(-self.eta * grad_img_y)

        loss_x = (grad_occ_x * weight_x).mean()
        loss_y = (grad_occ_y * weight_y).mean()

        return loss_x + loss_y


class FullLoss(nn.Module):
    def __init__(self, alpha_ssim=1.0, alpha_smd=0.1, alpha_smo=0.05, eta=100.0):
        super(FullLoss, self).__init__()

        self.wpm_loss = WeightedPhotometricLoss()
        self.rec_loss = ReconstructionLoss()
        self.ssim_loss = SSIMLoss()
        self.smd_loss = SmoothnessLoss(eta)
        self.smo_loss = OcclusionSmoothnessLoss(eta)

        self.alpha_ssim = alpha_ssim
        self.alpha_smd = alpha_smd
        self.alpha_smo = alpha_smo

    def forward(self, img_center, img_left, img_right,
                coarse_disparity, refined_disparity,
                occ_left, occ_right, use_coarse=False):

        if use_coarse:
            disparity = coarse_disparity
        else:
            disparity = refined_disparity

        print(f"[FullLoss DEBUG] img_center: {img_center.shape}, disparity: {disparity.shape}, dim: {disparity.dim()}")

        if disparity.dim() == 3:
            disparity = disparity.unsqueeze(1)
            print(f"[FullLoss DEBUG] After unsqueeze disparity: {disparity.shape}")

        warped_left = warp_image(img_left, disparity)
        warped_right = warp_image(img_right, -disparity)

        l_wpm = self.wpm_loss(img_left, img_right, img_center, occ_left, occ_right)
        l_rec = self.rec_loss(img_center, warped_left, warped_right, occ_left, occ_right)

        l_ssim = (self.ssim_loss(warped_left, img_center) + self.ssim_loss(warped_right, img_center)) / 2

        if disparity.shape[2] != img_center.shape[2] or disparity.shape[3] != img_center.shape[3]:
            print(f"[FullLoss DEBUG] Interpolating disparity from {disparity.shape[2:]} to {img_center.shape[2:]}")
            disparity_for_smd = F.interpolate(disparity, size=img_center.shape[2:], mode='bilinear', align_corners=False)
        else:
            disparity_for_smd = disparity

        l_smd = self.smd_loss(disparity_for_smd, img_center)
        l_smo = self.smo_loss(occ_left, img_center)

        loss = l_wpm + l_rec + self.alpha_ssim * l_ssim + self.alpha_smd * l_smd + self.alpha_smo * l_smo

        return loss, {
            'wpm': l_wpm.item(),
            'rec': l_rec.item(),
            'ssim': l_ssim.item(),
            'smd': l_smd.item(),
            'smo': l_smo.item(),
            'total': loss.item()
        }


def test_losses():
    B, C, H, W = 2, 3, 256, 256

    img_center = torch.randn(B, C, H, W)
    img_left = torch.randn(B, C, H, W)
    img_right = torch.randn(B, C, H, W)

    coarse_disparity = torch.randn(B, 1, H, W)
    refined_disparity = torch.randn(B, 1, H, W)

    occ_left = torch.rand(B, 1, H, W)
    occ_right = 1 - occ_left

    criterion = FullLoss(alpha_ssim=1.0, alpha_smd=0.1, alpha_smo=0.05, eta=100.0)

    loss, loss_dict = criterion(img_center, img_left, img_right,
                                coarse_disparity, refined_disparity,
                                occ_left, occ_right)

    print("Loss values:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")

    return criterion


if __name__ == "__main__":
    test_losses()
