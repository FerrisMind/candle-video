#!/usr/bin/env python3
"""
Generate Conv3d reference data for parity testing.

Generates test data for various Conv3d configurations:
- Standard Conv3d (non-causal)
- Causal Conv3d (LTX-Video style)
- Causal Conv3d (Wan style)
- Various kernel sizes: (3,3,3), (1,1,1), (3,1,1), (1,3,3)
- Various strides and dilations
- Grouped convolution

Reference data format (gen_conv3d_ref.safetensors):
  - input: (B, C, T, H, W)
  - weight: (out_c, in_c/g, kt, kh, kw)
  - bias: (out_c,) or None
  - output_standard: Conv3d output (non-causal)
  - output_causal: CausalConv3d output
  - config: {kernel, stride, padding, dilation, groups, is_causal}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from typing import Tuple, Dict, Any, Optional
import json


def create_standard_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> nn.Conv3d:
    """Create standard PyTorch Conv3d."""
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=True,
    )


class LTXStyleCausalConv3d(nn.Module):
    """LTX-Video style causal Conv3d with replicate padding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        dilation: Tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
        is_causal: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.is_causal = is_causal
        
        # Spatial padding only (temporal handled manually)
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, height_pad, width_pad),
            dilation=dilation,
            groups=groups,
            bias=True,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kt = self.kernel_size[0]
        
        if self.is_causal:
            # Causal: pad left only with replicated first frame
            pad_left = x[:, :, :1, :, :].repeat(1, 1, kt - 1, 1, 1)
            x = torch.cat([pad_left, x], dim=2)
        else:
            # Non-causal: symmetric padding
            pad_left = x[:, :, :1, :, :].repeat(1, 1, (kt - 1) // 2, 1, 1)
            pad_right = x[:, :, -1:, :, :].repeat(1, 1, (kt - 1) // 2, 1, 1)
            x = torch.cat([pad_left, x, pad_right], dim=2)
        
        return self.conv(x)


class WanStyleCausalConv3d(nn.Conv3d):
    """Wan style causal Conv3d with F.pad."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        # Wan uses F.pad with (w_left, w_right, h_top, h_bottom, t_front, t_back)
        # Causal: 2*padding[0] on front, 0 on back
        self._padding = (
            self.padding[2], self.padding[2],  # width
            self.padding[1], self.padding[1],  # height
            2 * self.padding[0], 0,            # time (causal: front only)
        )
        self.padding = (0, 0, 0)
    
    def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding, mode='replicate')
        return super().forward(x)


def generate_test_case(
    name: str,
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int, int],
    input_shape: Tuple[int, int, int, int, int],  # (B, C, T, H, W)
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
    is_causal: bool = False,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate a single test case with input, weights, and outputs."""
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create input
    x = torch.randn(*input_shape, device=device, dtype=torch.float32)
    
    # Create weight and bias
    kt, kh, kw = kernel_size
    weight = torch.randn(out_channels, in_channels // groups, kt, kh, kw, device=device, dtype=torch.float32) * 0.1
    bias = torch.randn(out_channels, device=device, dtype=torch.float32) * 0.01
    
    results = {
        f"{name}_input": x.cpu(),
        f"{name}_weight": weight.cpu(),
        f"{name}_bias": bias.cpu(),
    }
    
    # Standard Conv3d output (with explicit padding)
    with torch.no_grad():
        # For standard conv, use symmetric padding
        ph, pw = kh // 2, kw // 2
        pt = kt // 2 if not is_causal else 0
        
        # Check if we can run standard conv (input must be >= kernel after padding)
        b, c, t, h, w = input_shape
        dt, dh, dw = dilation
        effective_kt = dt * (kt - 1) + 1
        effective_kh = dh * (kh - 1) + 1
        effective_kw = dw * (kw - 1) + 1
        
        can_run_standard = (
            (t + 2 * pt) >= effective_kt and
            (h + 2 * ph) >= effective_kh and
            (w + 2 * pw) >= effective_kw
        )
        
        if can_run_standard:
            standard_conv = create_standard_conv3d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=(pt, ph, pw),
                dilation=dilation, groups=groups
            ).to(device)
            standard_conv.weight.data = weight
            standard_conv.bias.data = bias
            
            output_standard = standard_conv(x)
            results[f"{name}_output_standard"] = output_standard.cpu()
        else:
            # Skip standard conv for cases where input is too small
            print(f"   (Skipping standard conv - input too small for kernel)")
    
    # Causal Conv3d output (LTX style)
    if is_causal:
        with torch.no_grad():
            causal_conv = LTXStyleCausalConv3d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, groups=groups,
                is_causal=True
            ).to(device)
            causal_conv.conv.weight.data = weight
            causal_conv.conv.bias.data = bias
            
            output_causal = causal_conv(x)
            results[f"{name}_output_causal"] = output_causal.cpu()
    
    return results


def generate_all_test_cases() -> Dict[str, torch.Tensor]:
    """Generate all test cases for Conv3d parity testing."""
    
    all_results = {}
    
    print("Generating Conv3d reference data...")
    print("=" * 60)
    
    # Test case 1: Basic 3x3x3 kernel, non-causal
    print("\n1. Basic 3x3x3 kernel (non-causal)")
    results = generate_test_case(
        name="basic_3x3x3",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        input_shape=(1, 4, 8, 16, 16),
        padding=(1, 1, 1),
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['basic_3x3x3_input'].shape}")
    print(f"   Output: {results['basic_3x3x3_output_standard'].shape}")
    
    # Test case 2: Basic 3x3x3 kernel, causal
    print("\n2. Basic 3x3x3 kernel (causal)")
    results = generate_test_case(
        name="causal_3x3x3",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        input_shape=(1, 4, 8, 16, 16),
        padding=(0, 1, 1),  # No temporal padding (handled by causal logic)
        is_causal=True,
    )
    all_results.update(results)
    print(f"   Input: {results['causal_3x3x3_input'].shape}")
    print(f"   Output causal: {results['causal_3x3x3_output_causal'].shape}")
    
    # Test case 3: Pointwise 1x1x1 kernel
    print("\n3. Pointwise 1x1x1 kernel")
    results = generate_test_case(
        name="pointwise_1x1x1",
        in_channels=4,
        out_channels=8,
        kernel_size=(1, 1, 1),
        input_shape=(1, 4, 8, 16, 16),
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['pointwise_1x1x1_input'].shape}")
    print(f"   Output: {results['pointwise_1x1x1_output_standard'].shape}")
    
    # Test case 4: Temporal-only 3x1x1 kernel
    print("\n4. Temporal-only 3x1x1 kernel")
    results = generate_test_case(
        name="temporal_3x1x1",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 1, 1),
        input_shape=(1, 4, 8, 16, 16),
        padding=(1, 0, 0),
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['temporal_3x1x1_input'].shape}")
    print(f"   Output: {results['temporal_3x1x1_output_standard'].shape}")
    
    # Test case 5: Spatial-only 1x3x3 kernel
    print("\n5. Spatial-only 1x3x3 kernel")
    results = generate_test_case(
        name="spatial_1x3x3",
        in_channels=4,
        out_channels=8,
        kernel_size=(1, 3, 3),
        input_shape=(1, 4, 8, 16, 16),
        padding=(0, 1, 1),
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['spatial_1x3x3_input'].shape}")
    print(f"   Output: {results['spatial_1x3x3_output_standard'].shape}")
    
    # Test case 6: Strided convolution
    print("\n6. Strided 3x3x3 convolution (stride=2)")
    results = generate_test_case(
        name="strided_3x3x3",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        input_shape=(1, 4, 8, 16, 16),
        stride=(2, 2, 2),
        padding=(1, 1, 1),
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['strided_3x3x3_input'].shape}")
    print(f"   Output: {results['strided_3x3x3_output_standard'].shape}")
    
    # Test case 7: Dilated convolution
    print("\n7. Dilated 3x3x3 convolution (dilation=2)")
    results = generate_test_case(
        name="dilated_3x3x3",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        input_shape=(1, 4, 8, 32, 32),  # Larger input for dilation
        dilation=(2, 2, 2),
        padding=(2, 2, 2),  # Adjusted for dilation
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['dilated_3x3x3_input'].shape}")
    print(f"   Output: {results['dilated_3x3x3_output_standard'].shape}")
    
    # Test case 8: Grouped convolution
    print("\n8. Grouped 3x3x3 convolution (groups=2)")
    results = generate_test_case(
        name="grouped_3x3x3",
        in_channels=8,
        out_channels=16,
        kernel_size=(3, 3, 3),
        input_shape=(1, 8, 8, 16, 16),
        padding=(1, 1, 1),
        groups=2,
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['grouped_3x3x3_input'].shape}")
    print(f"   Output: {results['grouped_3x3x3_output_standard'].shape}")
    
    # Test case 9: Single frame input (edge case)
    print("\n9. Single frame input (T=1)")
    results = generate_test_case(
        name="single_frame",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        input_shape=(1, 4, 1, 16, 16),
        padding=(1, 1, 1),
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['single_frame_input'].shape}")
    print(f"   Output: {results['single_frame_output_standard'].shape}")
    
    # Test case 10: Single frame causal
    print("\n10. Single frame causal (T=1)")
    results = generate_test_case(
        name="single_frame_causal",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        input_shape=(1, 4, 1, 16, 16),
        padding=(0, 1, 1),
        is_causal=True,
    )
    all_results.update(results)
    print(f"   Input: {results['single_frame_causal_input'].shape}")
    print(f"   Output causal: {results['single_frame_causal_output_causal'].shape}")
    
    # Test case 11: Larger batch size
    print("\n11. Batch size > 1 (B=2)")
    results = generate_test_case(
        name="batch_2",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3, 3),
        input_shape=(2, 4, 8, 16, 16),
        padding=(1, 1, 1),
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['batch_2_input'].shape}")
    print(f"   Output: {results['batch_2_output_standard'].shape}")
    
    # Test case 12: LTX-Video VAE style (1024 -> 4096 channels, 3x3x3)
    print("\n12. LTX-Video VAE style (1024 -> 4096, 3x3x3)")
    results = generate_test_case(
        name="ltx_vae_style",
        in_channels=1024,
        out_channels=4096,
        kernel_size=(3, 3, 3),
        input_shape=(1, 1024, 4, 8, 8),  # Smaller spatial for memory
        padding=(0, 1, 1),
        is_causal=True,
        seed=123,
    )
    all_results.update(results)
    print(f"   Input: {results['ltx_vae_style_input'].shape}")
    print(f"   Output causal: {results['ltx_vae_style_output_causal'].shape}")
    
    # Test case 13: Wan VAE style (16 -> 512 channels, 3x3x3)
    print("\n13. Wan VAE style (16 -> 512, 3x3x3)")
    results = generate_test_case(
        name="wan_vae_style",
        in_channels=16,
        out_channels=512,
        kernel_size=(3, 3, 3),
        input_shape=(1, 16, 5, 32, 32),
        padding=(0, 1, 1),
        is_causal=True,
        seed=456,
    )
    all_results.update(results)
    print(f"   Input: {results['wan_vae_style_input'].shape}")
    print(f"   Output causal: {results['wan_vae_style_output_causal'].shape}")
    
    # Test case 14: Asymmetric kernel (3, 1, 1) causal
    print("\n14. Asymmetric kernel (3,1,1) causal")
    results = generate_test_case(
        name="asymmetric_3x1x1_causal",
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 1, 1),
        input_shape=(1, 4, 8, 16, 16),
        padding=(0, 0, 0),
        is_causal=True,
    )
    all_results.update(results)
    print(f"   Input: {results['asymmetric_3x1x1_causal_input'].shape}")
    print(f"   Output causal: {results['asymmetric_3x1x1_causal_output_causal'].shape}")
    
    # Test case 15: Depthwise convolution (groups = in_channels)
    print("\n15. Depthwise 3x3x3 convolution (groups=in_channels)")
    results = generate_test_case(
        name="depthwise_3x3x3",
        in_channels=8,
        out_channels=8,
        kernel_size=(3, 3, 3),
        input_shape=(1, 8, 8, 16, 16),
        padding=(1, 1, 1),
        groups=8,
        is_causal=False,
    )
    all_results.update(results)
    print(f"   Input: {results['depthwise_3x3x3_input'].shape}")
    print(f"   Output: {results['depthwise_3x3x3_output_standard'].shape}")
    
    return all_results


def save_config_metadata(results: Dict[str, torch.Tensor], output_path: str):
    """Save configuration metadata as JSON alongside safetensors."""
    config = {
        "test_cases": [
            {"name": "basic_3x3x3", "kernel": [3,3,3], "stride": [1,1,1], "padding": [1,1,1], "groups": 1, "is_causal": False},
            {"name": "causal_3x3x3", "kernel": [3,3,3], "stride": [1,1,1], "padding": [0,1,1], "groups": 1, "is_causal": True},
            {"name": "pointwise_1x1x1", "kernel": [1,1,1], "stride": [1,1,1], "padding": [0,0,0], "groups": 1, "is_causal": False},
            {"name": "temporal_3x1x1", "kernel": [3,1,1], "stride": [1,1,1], "padding": [1,0,0], "groups": 1, "is_causal": False},
            {"name": "spatial_1x3x3", "kernel": [1,3,3], "stride": [1,1,1], "padding": [0,1,1], "groups": 1, "is_causal": False},
            {"name": "strided_3x3x3", "kernel": [3,3,3], "stride": [2,2,2], "padding": [1,1,1], "groups": 1, "is_causal": False},
            {"name": "dilated_3x3x3", "kernel": [3,3,3], "stride": [1,1,1], "padding": [2,2,2], "dilation": [2,2,2], "groups": 1, "is_causal": False},
            {"name": "grouped_3x3x3", "kernel": [3,3,3], "stride": [1,1,1], "padding": [1,1,1], "groups": 2, "is_causal": False},
            {"name": "single_frame", "kernel": [3,3,3], "stride": [1,1,1], "padding": [1,1,1], "groups": 1, "is_causal": False},
            {"name": "single_frame_causal", "kernel": [3,3,3], "stride": [1,1,1], "padding": [0,1,1], "groups": 1, "is_causal": True},
            {"name": "batch_2", "kernel": [3,3,3], "stride": [1,1,1], "padding": [1,1,1], "groups": 1, "is_causal": False},
            {"name": "ltx_vae_style", "kernel": [3,3,3], "stride": [1,1,1], "padding": [0,1,1], "groups": 1, "is_causal": True},
            {"name": "wan_vae_style", "kernel": [3,3,3], "stride": [1,1,1], "padding": [0,1,1], "groups": 1, "is_causal": True},
            {"name": "asymmetric_3x1x1_causal", "kernel": [3,1,1], "stride": [1,1,1], "padding": [0,0,0], "groups": 1, "is_causal": True},
            {"name": "depthwise_3x3x3", "kernel": [3,3,3], "stride": [1,1,1], "padding": [1,1,1], "groups": 8, "is_causal": False},
        ],
        "format": {
            "input": "(B, C, T, H, W)",
            "weight": "(out_c, in_c/groups, kt, kh, kw)",
            "bias": "(out_c,)",
            "output_standard": "Conv3d output (non-causal or with symmetric padding)",
            "output_causal": "CausalConv3d output (causal mode only)",
        },
        "tolerance": {
            "f32": 1e-5,
            "bf16": 1e-3,
            "f16": 1e-3,
        }
    }
    
    config_path = output_path.replace(".safetensors", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config to {config_path}")


def main():
    print("=" * 60)
    print("Conv3d Reference Data Generator")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Generate all test cases
    results = generate_all_test_cases()
    
    # Save to safetensors
    output_path = "gen_conv3d_ref.safetensors"
    save_file(results, output_path)
    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} tensors to {output_path}")
    
    # Save config metadata
    save_config_metadata(results, output_path)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary of saved tensors:")
    print(f"{'=' * 60}")
    for name, tensor in sorted(results.items()):
        print(f"  {name}: {list(tensor.shape)}, dtype={tensor.dtype}")
    
    print(f"\n{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
