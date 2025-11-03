# Copyright 2024-2025 VGI Lab. All rights reserved.
"""
Ovi Text-Video Cross Attention Visualization Utilities

타임스텝별 Text-Video Cross Attention Map을 저장하고 시각화합니다.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2


class OviAttentionStore:
    """
    Ovi 모델의 Text-Video Cross Attention을 타임스텝별로 저장

    저장 구조:
    {
        timestep: {
            layer_idx: {
                'weights': torch.Tensor,  # Attention weights
                'q_shape': tuple,  # Query shape
                'k_shape': tuple   # Key shape
            }
        }
    }
    """

    def __init__(self, token_idx: Optional[int] = None):
        # Final storage: {timestep: {'spatial_map': [Lq], 'grid_sizes': ...}}
        self.attention_maps = {}
        self.current_timestep = None
        self.enabled = False
        self.token_idx = token_idx  # Which token to visualize

        # Temporary buffer for current timestep (running average)
        self.temp_spatial_sum = None  # Accumulate spatial map [Lq] across layers
        self.temp_layer_count = 0
        self.temp_grid_sizes = None

    def set_timestep(self, t: int):
        """현재 타임스텝 설정 - 이전 타임스텝 결과 저장하고 새로 시작"""
        # Save previous timestep result
        if self.current_timestep is not None and self.temp_spatial_sum is not None:
            # Compute average across layers
            avg_spatial_map = self.temp_spatial_sum / max(self.temp_layer_count, 1)
            self.attention_maps[self.current_timestep] = {
                'spatial_map': avg_spatial_map,  # [Lq]
                'grid_sizes': self.temp_grid_sizes
            }

        # Reset for new timestep
        self.current_timestep = t
        self.temp_spatial_sum = None
        self.temp_layer_count = 0
        self.temp_grid_sizes = None

    def enable(self):
        """Attention 저장 활성화"""
        self.enabled = True
        print("[OviAttentionStore] Attention storage enabled")

    def disable(self):
        """Attention 저장 비활성화"""
        self.enabled = False
        print("[OviAttentionStore] Attention storage disabled")

    def store(self, attn_weights: torch.Tensor, q_shape: tuple, k_shape: tuple, grid_sizes: torch.Tensor):
        """
        Attention weights를 spatial map으로 축소하여 running average로 누적

        Args:
            attn_weights: Full attention weights [B, H, Lq, Lk]
            q_shape: Query tensor shape (unused, kept for API compatibility)
            k_shape: Key tensor shape (unused, kept for API compatibility)
            grid_sizes: Grid sizes for spatial reconstruction [B, F, 2] (height, width per frame)
        """
        if not self.enabled or self.current_timestep is None:
            return

        # Extract spatial attention map [Lq] from full attention [B, H, Lq, Lk]
        # This is the KEY optimization: reduce [B,H,Lq,Lk] -> [Lq] immediately!
        Lk = attn_weights.shape[3]

        if self.token_idx is not None:
            # Specific token: attn_weights[0, :, :, token_idx] -> [H, Lq]
            if self.token_idx < Lk:
                spatial_map = attn_weights[0, :, :, self.token_idx].mean(dim=0)  # [Lq]
            else:
                print(f"[Warning] token_idx {self.token_idx} out of range, using average")
                spatial_map = attn_weights[0].mean(dim=(0, 2))  # [Lq]
        else:
            # All tokens average: mean over H and Lk dimensions
            spatial_map = attn_weights[0].mean(dim=(0, 2))  # [Lq]

        # Move to CPU immediately to free GPU memory
        spatial_map_cpu = spatial_map.detach().cpu()

        # Initialize or accumulate
        if self.temp_spatial_sum is None:
            self.temp_spatial_sum = spatial_map_cpu
            self.temp_grid_sizes = grid_sizes.detach().cpu() if grid_sizes is not None else None
        else:
            self.temp_spatial_sum += spatial_map_cpu

        self.temp_layer_count += 1

    def get_attention_for_timestep(self, timestep: int) -> Optional[Dict]:
        """
        특정 타임스텝의 attention 가져오기 (레이어 평균화된 결과)

        Args:
            timestep: 타임스텝

        Returns:
            Dictionary with 'weights' [B, H, Lq, Lk] and 'grid_sizes', or None
        """
        if timestep not in self.attention_maps:
            return None

        return self.attention_maps[timestep]

    def get_all_timesteps(self) -> List[int]:
        """저장된 모든 타임스텝 반환"""
        return sorted(self.attention_maps.keys(), reverse=True)

    def finalize(self):
        """마지막 timestep 저장 (생성 완료 후 호출)"""
        if self.current_timestep is not None and self.temp_spatial_sum is not None:
            avg_spatial_map = self.temp_spatial_sum / max(self.temp_layer_count, 1)
            self.attention_maps[self.current_timestep] = {
                'spatial_map': avg_spatial_map,
                'grid_sizes': self.temp_grid_sizes
            }
            print(f"[OviAttentionStore] Finalized timestep {self.current_timestep} with {self.temp_layer_count} layers")

    def clear(self):
        """모든 저장된 데이터 삭제"""
        self.attention_maps.clear()
        self.current_timestep = None
        self.temp_spatial_sum = None
        self.temp_layer_count = 0
        self.temp_grid_sizes = None
        print("[OviAttentionStore] All stored attention maps cleared")


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = True,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    히트맵(mask)을 원본 이미지(img)에 오버레이합니다.

    Args:
        img: 원본 이미지 [H, W, 3] (0-255 또는 0-1)
        mask: Attention mask [H, W] (0-1)
        use_rgb: RGB 형식 사용 여부
        colormap: OpenCV colormap

    Returns:
        오버레이된 이미지 [H, W, 3] (0-255)
    """
    # 1. 마스크를 [0, 255] 범위의 8비트 이미지로 변환
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 2. 이미지와 히트맵 합성
    heatmap = np.float32(heatmap) / 255

    if img is None:
        cam = heatmap
    else:
        # 이미지 정규화
        if img.max() > 1:
            img = np.float32(img) / 255
        else:
            img = np.float32(img)
        cam = heatmap * 0.5 + img * 0.5

    # 3. [0, 1] 범위를 [0, 255]로 복구
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def visualize_attention_map(
    attn_data: Dict,
    frame: Image.Image,
    frame_idx: int,
    text_tokens: List[str],
    token_idx: Optional[int] = None,
    resize_to: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Spatial attention map을 비디오 프레임에 시각화

    Args:
        attn_data: Dictionary with 'spatial_map' [Lq], 'grid_sizes'
        frame: 비디오 프레임
        frame_idx: 시각화할 프레임 인덱스
        text_tokens: 텍스트 토큰 리스트 (unused, kept for compatibility)
        token_idx: 특정 토큰 인덱스 (unused, already applied in store())
        resize_to: 출력 이미지 크기 (H, W)

    Returns:
        시각화된 이미지
    """
    spatial_attn = attn_data['spatial_map']  # [Lq]
    grid_sizes = attn_data.get('grid_sizes')  # [B, 3] where 3 = (F, H, W)

    # Reshape to spatial grid
    # spatial_attn: [Lq] = all frames' patches (F * H * W)
    # grid_sizes: [B, 3] where 3 = (num_frames, height_patches, width_patches)
    Lq = len(spatial_attn)

    # Extract patch dimensions from grid_sizes
    if grid_sizes is not None:
        if len(grid_sizes.shape) == 2 and grid_sizes.shape[1] == 3:
            # grid_sizes is [B, 3] = [B, (F, H, W)]
            num_frames, h_patches, w_patches = grid_sizes[0].tolist()
            num_frames, h_patches, w_patches = int(num_frames), int(h_patches), int(w_patches)
        elif len(grid_sizes.shape) == 1 and grid_sizes.shape[0] == 3:
            # grid_sizes is [3] = (F, H, W)
            num_frames, h_patches, w_patches = grid_sizes.tolist()
            num_frames, h_patches, w_patches = int(num_frames), int(h_patches), int(w_patches)
        else:
            # Fallback: default 512x992 resolution = 32x62 patches, 31 frames
            print(f"[Warning] Unexpected grid_sizes shape: {grid_sizes.shape}, using fallback")
            num_frames, h_patches, w_patches = 31, 32, 62
    else:
        # Fallback: default 512x992 resolution = 32x62 patches, 31 frames
        num_frames, h_patches, w_patches = 31, 32, 62

    # Calculate patches per frame
    patches_per_frame = h_patches * w_patches

    # Extract attention for target frame
    start_idx = frame_idx * patches_per_frame
    end_idx = start_idx + patches_per_frame

    if end_idx <= Lq:
        frame_attn = spatial_attn[start_idx:end_idx]  # [H*W]

        # Convert to float32 for numpy compatibility (BFloat16 not supported)
        frame_attn = frame_attn.float()

        # Reshape to 2D
        attn_map_2d = frame_attn.reshape(h_patches, w_patches).numpy()
    else:
        # Fallback: use mean
        print(f"[Warning] Attention range out of bounds: {end_idx} > {Lq}, using fallback")
        attn_map_2d = np.ones((h_patches, w_patches)) * spatial_attn.float().mean().item()

    # 4. Resize to frame size
    attn_map_resized = cv2.resize(attn_map_2d, (frame.width, frame.height))

    # 5. Normalize (0-1)
    attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)

    # 6. Apply to frame
    frame_np = np.array(frame)
    vis = show_cam_on_image(frame_np, attn_map_resized, use_rgb=True)

    # 7. Resize if needed
    if resize_to is not None:
        vis = cv2.resize(vis, (resize_to[1], resize_to[0]))

    return Image.fromarray(vis)


def create_attention_video(
    attention_store: OviAttentionStore,
    video_frames: List[Image.Image],
    text_tokens: List[str],
    token_idx: Optional[int] = None,
    output_path: str = "attention_video.gif",
    fps: int = 10,
    frame_idx: int = 0
):
    """
    타임스텝별 attention map으로 비디오 생성

    Args:
        attention_store: OviAttentionStore 인스턴스
        video_frames: 생성된 비디오 프레임들
        text_tokens: 텍스트 토큰 리스트
        token_idx: 시각화할 토큰 인덱스 (None이면 전체 평균)
        output_path: 출력 파일 경로 (.gif 또는 .mp4)
        fps: 프레임 레이트
        frame_idx: 시각화할 비디오 프레임 인덱스 (기본값: 0)
    """
    # Finalize last timestep
    attention_store.finalize()

    timesteps = attention_store.get_all_timesteps()

    if not timesteps:
        print("[Error] No attention maps stored!")
        return

    print(f"[OviAttentionViz] Creating attention video with {len(timesteps)} timesteps...")

    vis_frames = []

    for t in timesteps:
        # 1. 해당 타임스텝의 attention 가져오기
        attn_data = attention_store.get_attention_for_timestep(t)

        if attn_data is None:
            continue

        # 2. 지정된 프레임에 시각화
        if video_frames and frame_idx < len(video_frames):
            frame = video_frames[frame_idx]

            # 3. 시각화
            vis_frame = visualize_attention_map(
                attn_data,
                frame,
                frame_idx,
                text_tokens,
                token_idx=token_idx
            )

            vis_frames.append(vis_frame)
        elif frame_idx >= len(video_frames):
            print(f"[Warning] Frame index {frame_idx} out of range (max: {len(video_frames)-1}). Using first frame.")

    if not vis_frames:
        print("[Error] No visualization frames generated!")
        return

    # 4. 비디오 저장
    if output_path.endswith('.gif'):
        # GIF 저장
        vis_frames[0].save(
            output_path,
            save_all=True,
            append_images=vis_frames[1:],
            duration=1000 // fps,
            loop=0
        )
        print(f"[OviAttentionViz] Saved GIF to {output_path}")
    else:
        # MP4 저장 (opencv 필요)
        height, width = vis_frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in vis_frames:
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"[OviAttentionViz] Saved MP4 to {output_path}")
