import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[3, 6, 8]):
        super(ASPP, self).__init__()

        self.convs = nn.ModuleList()

        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        for rate in atrous_rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d((len(self.convs) + 1) * out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        res = []

        for conv in self.convs:
            res.append(conv(x))

        if x.shape[2] > 1 and x.shape[3] > 1:
            gap = self.global_avg_pool(x)
            gap = torch.nn.functional.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
            res.append(gap)
        else:
            res.append(x)

        res = torch.cat(res, dim=1)
        return self.project(res)


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, max_channels=128):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )

        self.aspp = ASPP(128, max_channels)

        self.out_channels = max_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.aspp(x)
        return x


class CostFilter3D(nn.Module):
    def __init__(self, in_channels, num_filters=2):
        super(CostFilter3D, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.filters = nn.ModuleList()

        for i in range(num_filters):
            layers = [
                nn.Conv3d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm3d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm3d(32),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            self.filters.append(nn.Sequential(*layers))

        self.output_conv = nn.Sequential(
            nn.Conv3d(32, 1, 1, 1, 0),
        )

    def forward(self, x):
        x = self.input_conv(x)
        for f in self.filters:
            x = f(x) + x

        return self.output_conv(x)


class DisparityRegression(nn.Module):
    def __init__(self, num_disparities):
        super(DisparityRegression, self).__init__()
        self.num_disparities = num_disparities

    def forward(self, cost):
        probs = F.softmax(cost, dim=1)
        disparities = torch.linspace(0, self.num_disparities - 1, self.num_disparities, device=cost.device)
        disparities = disparities.view(1, self.num_disparities, 1, 1).repeat(1, 1, cost.shape[2], cost.shape[3])

        disparity = torch.sum(probs * disparities, dim=1)
        return disparity


class DispNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 max_channels=128,
                 num_disparities=25,
                 num_residual_disparities=21,
                 num_cost_filters=2):
        super(DispNet, self).__init__()

        self.feature_extractor = FeatureExtractor(in_channels, max_channels)

        self.coarse_cost_filter = CostFilter3D(max_channels, num_cost_filters)
        self.residual_cost_filter = CostFilter3D(max_channels, num_cost_filters)

        self.coarse_disparity_regression = DisparityRegression(num_disparities)
        self.residual_disparity_regression = DisparityRegression(num_residual_disparities)

        self.num_disparities = num_disparities
        self.num_residual_disparities = num_residual_disparities

        self.coarse_range = 24
        self.residual_range = 2.0

        self.refine_conv = nn.Sequential(
            nn.Conv2d(1 + max_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def build_cost_volume(self, ref_features, src_features, disparity_range, num_disparities):
        B, C, H, W = ref_features.shape
        device = ref_features.device

        disparity_samples = torch.linspace(disparity_range[0], disparity_range[1],
                                          num_disparities, device=device)
        disparity_samples = disparity_samples.view(1, num_disparities, 1, 1)

        cost_volume = []
        for d in range(num_disparities):
            d_val = disparity_samples[0, d, 0, 0]

            shifted_src = torch.roll(src_features, shifts=int(-d_val), dims=3)
            shifted_src[:, :, :, :int(d_val)] = 0

            diff = (ref_features - shifted_src) ** 2
            cost_volume.append(diff)

        cost_volume = torch.stack(cost_volume, dim=2)
        return cost_volume

    def variance_feature_matching(self, ref_features, src_features_left, src_features_right,
                                   disparity_range, num_disparities):
        B, C, H, W = ref_features.shape
        device = ref_features.device

        disparity_samples = torch.linspace(disparity_range[0], disparity_range[1],
                                          num_disparities, device=device)
        disparity_samples = disparity_samples.view(1, num_disparities, 1, 1)

        cost_volume = []
        for d in range(num_disparities):
            d_val = disparity_samples[0, d, 0, 0]

            shifted_left = torch.roll(src_features_left, shifts=int(-d_val), dims=3)
            shifted_left[:, :, :, :int(d_val)] = 0

            shifted_right = torch.roll(src_features_right, shifts=int(d_val), dims=3)
            shifted_right[:, :, :, -int(d_val):] = 0

            features = torch.stack([ref_features, shifted_left, shifted_right], dim=0)
            variance = torch.var(features, dim=0, unbiased=False)

            cost_volume.append(variance)

        cost_volume = torch.stack(cost_volume, dim=2)
        return cost_volume

    def forward(self, img_center, img_left, img_right):
        feat_c = self.feature_extractor(img_center)
        feat_l = self.feature_extractor(img_left)
        feat_r = self.feature_extractor(img_right)

        coarse_range = (-self.coarse_range, self.coarse_range)
        cost_volume = self.variance_feature_matching(feat_c, feat_l, feat_r,
                                                      coarse_range, self.num_disparities)

        coarse_cost = self.coarse_cost_filter(cost_volume)
        coarse_cost = coarse_cost.squeeze(1)

        coarse_disparity = self.coarse_disparity_regression(coarse_cost)

        residual_range = (-self.residual_range, self.residual_range)
        coarse_d_expanded = coarse_disparity.unsqueeze(1)

        cost_volume_res = self.variance_feature_matching(feat_c, feat_l, feat_r,
                                                         residual_range, self.num_residual_disparities)

        residual_cost = self.residual_cost_filter(cost_volume_res)
        residual_cost = residual_cost.squeeze(1)

        residual_disparity = self.residual_disparity_regression(residual_cost)

        refined_disparity = coarse_disparity + residual_disparity * (self.residual_range / self.num_residual_disparities * 2)

        return coarse_disparity, refined_disparity, feat_c


def test_dispnet():
    model = DispNet(
        in_channels=3,
        max_channels=128,
        num_disparities=25,
        num_residual_disparities=21,
        num_cost_filters=2
    )

    B, C, H, W = 2, 3, 256, 256
    img_center = torch.randn(B, C, H, W)
    img_left = torch.randn(B, C, H, W)
    img_right = torch.randn(B, C, H, W)

    coarse_d, refined_d, features = model(img_center, img_left, img_right)

    print(f"Input shape: {img_center.shape}")
    print(f"Coarse disparity shape: {coarse_d.shape}")
    print(f"Refined disparity shape: {refined_d.shape}")
    print(f"Features shape: {features.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.3f}M")

    return model


if __name__ == "__main__":
    test_dispnet()
