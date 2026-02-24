import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import glob
import json

from models.dispnet import DispNet
from utils.losses import warp_image


def load_light_field(data_dir, scene_name, angular_size=7):
    lf_path = os.path.join(data_dir, scene_name)

    subfolders = sorted([d for d in os.listdir(lf_path) if os.path.isdir(os.path.join(lf_path, d))])

    if len(subfolders) < angular_size:
        print(f"Warning: Not enough angular views in {scene_name}")
        return None

    lf = []
    center = angular_size // 2

    for u in range(angular_size):
        row_imgs = []
        for v in range(angular_size):
            img_path = os.path.join(lf_path, subfolders[u], f"{v:02d}.png")

            if not os.path.exists(img_path):
                img_path = os.path.join(lf_path, subfolders[u], f"{v}.png")

            if not os.path.exists(img_path):
                img_path = os.path.join(lf_path, f"view_{u}_{v}.png")

            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = np.array(img, dtype=np.float32) / 255.0
                row_imgs.append(torch.from_numpy(img).permute(2, 0, 1))
            else:
                return None

        lf.append(torch.stack(row_imgs, dim=0))

    lf = torch.stack(lf, dim=0)

    return lf


def get_view_combinations(angular_size=7, dense=True):
    center = angular_size // 2
    combinations = []

    if dense:
        offsets = [2, 3]
    else:
        offsets = [1]

    for offset in offsets:
        if center - offset >= 0 and center + offset < angular_size:
            combinations.append((center - offset, center, center + offset, offset, False))

        if center - offset >= 0 and center + offset < angular_size:
            combinations.append((center, center - offset, center + offset, offset, True))

    return combinations


def estimate_disparity(dispnet, img_center, img_left, img_right, device):
    dispnet.eval()

    with torch.no_grad():
        img_center = img_center.unsqueeze(0).to(device)
        img_left = img_left.unsqueeze(0).to(device)
        img_right = img_right.unsqueeze(0).to(device)

        coarse_disparity, refined_disparity, features = dispnet(img_center, img_left, img_right)

        disparity = refined_disparity.squeeze(0).squeeze(0).cpu().numpy()

    return disparity


def compute_warping_error(dispnet, img_center, img_source, disparity, device):
    with torch.no_grad():
        img_center = img_center.unsqueeze(0).to(device)
        img_source = img_source.unsqueeze(0).to(device)
        disparity = disparity.unsqueeze(0).unsqueeze(0).to(device)

        warped = warp_image(img_source, disparity)

        error = torch.abs(warped - img_center).squeeze(0).mean(0).cpu().numpy()

    return error


def fuse_disparities(disparity_maps, error_maps, quantile=0.95, n_prime=2):
    """
    Fuse multiple disparity maps using weighted fusion based on estimated errors.
    """
    H, W = disparity_maps[0].shape

    final_disparity = np.zeros((H, W))

    for y in range(H):
        for x in range(W):
            errors = [error_maps[j][y, x] for j in range(len(disparity_maps))]

            sorted_indices = np.argsort(errors)[:n_prime]

            min_error = errors[sorted_indices[0]]
            max_error = errors[sorted_indices[-1]]

            if max_error > min_error + 1e-6:
                weights = np.array([np.exp(-errors[i]) for i in sorted_indices])
                weights = weights / weights.sum()
            else:
                weights = np.ones(n_prime) / n_prime

            fused_value = sum(weights[i] * disparity_maps[sorted_indices[i]][y, x] for i in range(n_prime))

            final_disparity[y, x] = fused_value

    return final_disparity


def inference_scene(dispnet, data_dir, scene_name, device, angular_size=7, dense=True):
    lf = load_light_field(data_dir, scene_name, angular_size)

    if lf is None:
        print(f"Could not load light field for {scene_name}")
        return None

    combinations = get_view_combinations(angular_size, dense)

    disparity_maps = []
    error_maps = []

    center = angular_size // 2

    for u_left, u_center, u_right, offset, is_vertical in combinations:
        if is_vertical:
            img_center = lf[u_center, center]
            img_left = lf[u_left, center]
            img_right = lf[u_right, center]
        else:
            img_center = lf[center, u_center]
            img_left = lf[center, u_left]
            img_right = lf[center, u_right]

        disparity = estimate_disparity(dispnet, img_center, img_left, img_right, device)

        if is_vertical:
            disparity = disparity

        disparity_maps.append(disparity)

        error_map = np.zeros_like(disparity)
        for aux_u in [center - 1, center + 1]:
            if 0 <= aux_u < angular_size:
                if is_vertical:
                    img_aux = lf[aux_u, center]
                else:
                    img_aux = lf[center, aux_u]

                error = compute_warping_error(dispnet, img_center, img_aux,
                                            torch.from_numpy(disparity), device)
                error_map += error

        error_map /= 2
        error_maps.append(error_map)

    if dense:
        final_disparity = fuse_disparities(disparity_maps, error_maps, n_prime=2)
    else:
        final_disparity = np.min(disparity_maps, axis=0)

    return final_disparity


def main():
    parser = argparse.ArgumentParser(description='Inference for Light Field Depth Estimation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to light field data')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--scene', type=str, default=None, help='Specific scene name (default: all scenes)')
    parser.add_argument('--angular_size', type=int, default=7, help='Angular size of light field')
    parser.add_argument('--dense', action='store_true', default=True, help='Use dense LF configuration')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    dispnet = DispNet(
        in_channels=3,
        max_channels=128,
        num_disparities=25,
        num_residual_disparities=21,
        num_cost_filters=2
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    dispnet.load_state_dict(checkpoint['dispnet_state_dict'])
    print(f"Loaded model from {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted([d for d in os.listdir(args.data_dir)
                       if os.path.isdir(os.path.join(args.data_dir, d))])

    results = []

    for scene_name in scenes:
        print(f"Processing {scene_name}...")

        disparity = inference_scene(dispnet, args.data_dir, scene_name, device,
                                  args.angular_size, args.dense)

        if disparity is not None:
            disp_normalized = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8)
            disp_img = (disp_normalized * 255).astype(np.uint8)

            output_path = os.path.join(args.output_dir, f"{scene_name}_disparity.png")
            Image.fromarray(disp_img).save(output_path)
            print(f"Saved disparity to {output_path}")

            results.append({
                'scene': scene_name,
                'output_path': output_path,
                'min_disparity': float(disparity.min()),
                'max_disparity': float(disparity.max())
            })

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
