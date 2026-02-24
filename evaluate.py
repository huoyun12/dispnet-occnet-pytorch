import os
import argparse
import torch
import numpy as np
from PIL import Image
import glob
import json

from models.dispnet import DispNet
from utils.losses import warp_image


def load_ground_truth(gt_dir, scene_name):
    gt_path = os.path.join(gt_dir, scene_name, 'depth.png')

    if os.path.exists(gt_path):
        depth = np.array(Image.open(gt_path), dtype=np.float32)
        if len(depth.shape) == 2:
            return depth
        elif len(depth.shape) == 3:
            return depth[:, :, 0]

    gt_path = os.path.join(gt_dir, scene_name, 'disparity.png')
    if os.path.exists(gt_path):
        disparity = np.array(Image.open(gt_path), dtype=np.float32)
        if len(disparity.shape) == 3:
            disparity = disparity[:, :, 0]
        return disparity

    return None


def load_light_field(data_dir, scene_name, angular_size=7):
    lf_path = os.path.join(data_dir, scene_name)

    subfolders = sorted([d for d in os.listdir(lf_path) if os.path.isdir(os.path.join(lf_path, d))])

    if len(subfolders) < angular_size:
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


def fuse_disparities(disparity_maps, error_maps, n_prime=2):
    H, W = disparity_maps[0].shape

    final_disparity = np.zeros((H, W))

    for y in range(H):
        for x in range(W):
            errors = [error_maps[j][y, x] for j in range(len(disparity_maps))]

            sorted_indices = np.argsort(errors)[:n_prime]

            weights = np.array([np.exp(-errors[i]) for i in sorted_indices])
            weights = weights / (weights.sum() + 1e-8)

            fused_value = sum(weights[i] * disparity_maps[sorted_indices[i]][y, x] for i in range(n_prime))

            final_disparity[y, x] = fused_value

    return final_disparity


def evaluate_scene(dispnet, data_dir, scene_name, device, angular_size=7, dense=True):
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


def compute_mse(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    return mse


def compute_bpr(pred, gt, threshold):
    diff = np.abs(pred - gt)
    bpr = np.mean(diff > threshold) * 100
    return bpr


def main():
    parser = argparse.ArgumentParser(description='Evaluate Light Field Depth Estimation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to light field data')
    parser.add_argument('--gt_dir', type=str, default=None, help='Path to ground truth (if available)')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate')
    parser.add_argument('--angular_size', type=int, default=7, help='Angular size of light field')
    parser.add_argument('--dense', action='store_true', default=True, help='Use dense LF configuration')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Output directory')

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

    scenes = sorted([d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))])

    results = []

    for scene_name in scenes:
        print(f"Evaluating {scene_name}...")

        disparity = evaluate_scene(dispnet, args.data_dir, scene_name, device,
                                 args.angular_size, args.dense)

        if disparity is None:
            continue

        result = {
            'scene': scene_name,
        }

        if args.gt_dir:
            gt = load_ground_truth(args.gt_dir, scene_name)

            if gt is not None:
                H_pred, W_pred = disparity.shape
                H_gt, W_gt = gt.shape

                if H_pred != H_gt or W_pred != W_gt:
                    disparity_resized = np.array(Image.fromarray(disparity).resize((W_gt, H_gt)))
                    gt_resized = gt
                else:
                    disparity_resized = disparity
                    gt_resized = gt

                mse = compute_mse(disparity_resized, gt_resized)

                bpr_007 = compute_bpr(disparity_resized, gt_resized, 0.07)
                bpr_003 = compute_bpr(disparity_resized, gt_resized, 0.03)
                bpr_001 = compute_bpr(disparity_resized, gt_resized, 0.01)

                result['mse'] = float(mse)
                result['bpr_0.07'] = float(bpr_007)
                result['bpr_0.03'] = float(bpr_003)
                result['bpr_0.01'] = float(bpr_001)

                print(f"  MSE: {mse:.4f}, BPR@0.07: {bpr_007:.2f}%, BPR@0.03: {bpr_003:.2f}%, BPR@0.01: {bpr_001:.2f}%")

        disp_normalized = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8)
        disp_img = (disp_normalized * 255).astype(np.uint8)

        output_path = os.path.join(args.output_dir, f"{scene_name}_disparity.png")
        Image.fromarray(disp_img).save(output_path)

        results.append(result)

    if len(results) > 0 and 'mse' in results[0]:
        avg_mse = np.mean([r['mse'] for r in results])
        avg_bpr_007 = np.mean([r['bpr_0.07'] for r in results])
        avg_bpr_003 = np.mean([r['bpr_0.03'] for r in results])
        avg_bpr_001 = np.mean([r['bpr_0.01'] for r in results])

        print(f"\nAverage Results:")
        print(f"  MSE: {avg_mse:.4f}")
        print(f"  BPR@0.07: {avg_bpr_007:.2f}%")
        print(f"  BPR@0.03: {avg_bpr_003:.2f}%")
        print(f"  BPR@0.01: {avg_bpr_001:.2f}%")

        summary = {
            'average_mse': float(avg_mse),
            'average_bpr_0.07': float(avg_bpr_007),
            'average_bpr_0.03': float(avg_bpr_003),
            'average_bpr_0.01': float(avg_bpr_001),
            'num_scenes': len(results)
        }
        results.append(summary)

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
