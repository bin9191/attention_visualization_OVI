# Ovi Attention Visualization Guide

본 문서는 Ovi 모델에 Text-to-Video Cross Attention 시각화 기능을 추가한 내용을 정리한 문서입니다.

## 📌 목차

1. [개요](#개요)
2. [새로 추가된 파일](#새로-추가된-파일)
3. [수정된 파일](#수정된-파일)
4. [핵심 기술](#핵심-기술)
5. [사용 방법](#사용-방법)
6. [문제 해결](#문제-해결)
7. [메모리 최적화](#메모리-최적화)

---

## 개요

Ovi 모델의 text-to-video 생성 과정에서 특정 텍스트 토큰(예: "man", "running")이 비디오 프레임의 어느 공간적 위치에 attention하는지를 시각화하는 기능입니다.

### 주요 기능

- ✅ **Flash Attention ↔ Standard Attention** config 기반 토글
- ✅ **특정 토큰 선택 시각화** (예: "man", "running", "park")
- ✅ **Timestep별 attention 변화** GIF로 생성 (50 timesteps)
- ✅ **메모리 최적화**: 20GB RAM에서 실행 가능 (타임스텝당 12.4MB)
- ✅ **자동 토큰 인덱스 로깅**: 어떤 인덱스가 어떤 단어인지 자동 표시
- ✅ **타임스탬프 파일명**: 생성 시각이 포함된 파일명

### 생성 결과물

```
A_man_running_at_the_park_512x992_103_0_20251102_193244.mp4
A_man_running_at_the_park_512x992_103_0_20251102_193244_attention_frame0_▁man.gif
```

- **MP4**: 121프레임의 생성된 비디오
- **GIF**: 50개 diffusion timestep에 걸친 attention map 히트맵 (빨간색 = 높은 attention)

---

## 새로 추가된 파일

### `ovi/utils/ovi_attention_viz.py`

Attention 저장 및 시각화를 위한 유틸리티 모듈입니다.

#### 주요 클래스: `OviAttentionStore`

Text-to-Video cross attention weights를 timestep별로 저장하고 관리합니다.

**핵심 메서드**:

```python
class OviAttentionStore:
    def __init__(self, token_idx: Optional[int] = None):
        """
        Args:
            token_idx: 시각화할 텍스트 토큰 인덱스 (None이면 전체 평균)
        """
        self.attention_maps = {}  # {timestep: {'spatial_map': [Lq], 'grid_sizes': ...}}
        self.token_idx = token_idx
        self.temp_spatial_sum = None  # Running average를 위한 임시 버퍼
        self.temp_layer_count = 0

    def store(self, attn_weights: torch.Tensor, q_shape, k_shape, grid_sizes):
        """
        Attention weights를 저장합니다.

        Args:
            attn_weights: [B, H, Lq, Lk] 형태의 attention weights
            q_shape: Query shape (video latent)
            k_shape: Key shape (text embeddings)
            grid_sizes: [B, 3] where 3=(F, H, W) - 프레임, 높이, 너비 패치 수

        메모리 최적화:
            1. GPU에서 즉시 spatial map [Lq] 추출
            2. CPU로 즉시 전송
            3. Running average 누적 (레이어별로 평균화)
        """

    def set_timestep(self, t: int):
        """
        새로운 timestep으로 전환합니다.
        이전 timestep의 누적된 attention을 평균화하여 저장합니다.
        """

    def finalize(self):
        """마지막 timestep의 attention을 저장합니다."""
```

**메모리 최적화 전략**:
- Full attention [B, H, Lq, Lk] 저장 대신 spatial map [Lq]만 추출
- GPU→CPU 즉시 전송으로 VRAM 절약
- Layer-wise running average로 RAM 절약

#### 주요 함수

**1. `visualize_attention_map()`**

```python
def visualize_attention_map(
    attn_data: dict,
    frame: Image.Image,
    frame_idx: int = 0
) -> Image.Image:
    """
    Spatial attention map을 2D 히트맵으로 시각화합니다.

    Args:
        attn_data: {'spatial_map': [Lq], 'grid_sizes': [B, 3]}
        frame: 원본 비디오 프레임 (PIL Image)
        frame_idx: 시각화할 프레임 인덱스

    Returns:
        Attention heatmap이 오버레이된 프레임

    처리 과정:
        1. grid_sizes에서 (F, H, W) 추출
        2. [Lq] → [H*W] 해당 프레임의 attention 추출
        3. [H*W] → [H, W] 2D로 reshape
        4. [H, W] → [frame_height, frame_width] 리사이즈
        5. Normalize & Colormap (빨강 = 높음, 파랑 = 낮음)
        6. 원본 프레임에 블렌딩 (alpha=0.4)
    """
```

**2. `create_attention_video()`**

```python
def create_attention_video(
    attention_store: OviAttentionStore,
    frame: Image.Image,
    output_path: str,
    token_name: str = "",
    frame_idx: int = 0,
    fps: int = 10
):
    """
    모든 timestep의 attention map을 GIF로 생성합니다.

    Args:
        attention_store: 저장된 attention maps
        frame: 원본 비디오 프레임
        output_path: GIF 저장 경로
        token_name: 토큰 이름 (표시용)
        frame_idx: 시각화할 프레임 인덱스
        fps: GIF 프레임레이트

    출력:
        50개 프레임의 GIF (각 diffusion timestep의 attention pattern)
    """
```

---

## 수정된 파일

### 1. `ovi/configs/inference/inference_fusion.yaml`

Config 파일에 attention 시각화 설정을 추가했습니다.

```yaml
# Flash Attention 설정
use_flash_attention: false  # false = 표준 attention 사용 (시각화 가능)
                            # true = Flash Attention 사용 (빠르지만 시각화 불가)

# Attention Visualization 설정
visualize_attention: true   # Attention 시각화 활성화
visualize_token_idx: 1      # 시각화할 토큰 인덱스
                            # 0 = "A", 1 = "man", 2 = "running", ...
                            # null = 전체 토큰의 평균
visualize_frame_idx: 0      # 시각화할 비디오 프레임 인덱스
                            # 0 = 첫 번째 프레임
```

**주의사항**:
- `use_flash_attention: true`일 때는 시각화가 불가능합니다 (Flash Attention은 weight를 저장하지 않음)
- 시각화를 원하면 반드시 `use_flash_attention: false`로 설정해야 합니다

---

### 2. `ovi/modules/attention.py`

Flash Attention과 Standard Attention을 config 기반으로 전환할 수 있도록 수정했습니다.

#### 추가된 전역 변수 및 함수

```python
# 전역 Flash Attention 토글
USE_FLASH_ATTENTION = True

def set_flash_attention_enabled(enabled: bool):
    """
    Flash Attention 사용 여부를 전역적으로 설정합니다.

    Args:
        enabled: True = Flash Attention, False = Standard Attention
    """
    global USE_FLASH_ATTENTION
    USE_FLASH_ATTENTION = enabled
    if not enabled:
        warnings.warn(
            "Flash attention is disabled. Using PyTorch standard attention instead. "
            "This may result in slower performance and higher memory usage."
        )
```

#### 수정된 `flash_attention()` 함수

```python
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Config에 따라 Flash Attention 또는 Standard Attention을 사용합니다.

    원래 코드는 Flash Attention만 사용했지만,
    시각화가 필요한 경우 Standard Attention으로 fallback합니다.
    """
    # Config에서 Flash Attention이 비활성화된 경우
    if not USE_FLASH_ATTENTION:
        return attention(q, k, v, k_lens=k_lens)

    # Flash Attention 2 또는 3 사용
    # ... (기존 코드 유지)
```

#### 수정된 `attention_with_weights()` 함수

```python
def attention_with_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_lens: Optional[torch.Tensor] = None,
    average_for_q: bool = True,  # 새로 추가된 파라미터
    total_video_latent_frames: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Attention weights를 함께 반환하는 함수입니다.

    Args:
        average_for_q: True면 query별 평균 attention 반환,
                      False면 전체 attention matrix 반환

    Returns:
        out: Attention 출력 [B, Lq, C]
        avg_attn_weights: 평균 attention weights
        attn_weights: 전체 attention weights [B, H, Lq, Lk]

    시각화를 위해 average_for_q=False로 설정하면
    [B, H, Lq, Lk] 형태의 전체 attention matrix를 반환합니다.
    """
```

---

### 3. `ovi/modules/fusion.py`

Fusion model의 forward pass에 attention 저장 기능을 추가했습니다.

#### 수정된 함수 시그니처

모든 forward 관련 함수에 `store_attention` 파라미터를 추가했습니다:

```python
# 1. FusionModel.forward()
def forward(self, ..., store_attention=False):
    """
    Args:
        store_attention: True이면 attention weights를 저장
    """

# 2. single_fusion_block_forward()
def single_fusion_block_forward(self, ..., store_attention=False):
    """Block 레벨에서 store_attention 플래그를 전달"""

# 3. single_fusion_cross_attention_ffn_forward()
def single_fusion_cross_attention_ffn_forward(self, ..., store_attention=False):
    """Cross attention 레이어로 플래그 전달"""

# 4. single_fusion_cross_attention_forward() - 핵심 수정
def single_fusion_cross_attention_forward(
    self,
    cross_attn_block,
    src_seq,
    src_grid_sizes,
    src_freqs,
    target_seq,
    target_seq_lens,
    target_grid_sizes,
    target_freqs,
    context,
    context_lens,
    store_attention=False  # 새로 추가
):
```

#### 핵심: Text-to-Video Cross Attention 저장 로직

```python
def single_fusion_cross_attention_forward(self, ..., store_attention=False):
    b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim

    # QKV projection
    if hasattr(cross_attn_block, "k_img"):
        q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
    else:
        q, k, v = cross_attn_block.qkv_fn(src_seq, context)
        k_img, v_img = None, None

    # ... (Sequence parallel 처리) ...

    # ========== Attention 계산 및 저장 ==========
    if store_attention and \
       hasattr(cross_attn_block, 'attention_store') and \
       cross_attn_block.attention_store is not None:

        # 시각화를 위해 attention_with_weights() 사용
        from .attention import attention_with_weights

        x, _, full_attn_weights = attention_with_weights(
            q, k, v,
            k_lens=context_lens,
            average_for_q=False,  # 전체 [B, H, Lq, Lk] 반환
            total_video_latent_frames=31
        )

        # AttentionStore에 저장
        if hasattr(cross_attn_block.attention_store, 'store'):
            cross_attn_block.attention_store.store(
                full_attn_weights,  # [B, H, Lq, Lk]
                q.shape,            # Query shape
                k.shape,            # Key shape
                src_grid_sizes      # [B, 3] = [B, (F, H, W)]
            )
    else:
        # 기본 모드: Flash Attention 사용 (빠름, weight 저장 안 함)
        x = flash_attention(q, k, v, k_lens=context_lens)

    # ... (나머지 cross attention 로직) ...
```

**주요 포인트**:
- `store_attention=True`일 때만 `attention_with_weights()` 사용
- `store_attention=False`일 때는 기존처럼 `flash_attention()` 사용
- `src_grid_sizes`를 함께 저장하여 spatial 정보 보존

---

### 4. `ovi/ovi_fusion_engine.py`

Diffusion loop에서 timestep을 AttentionStore에 전달하도록 수정했습니다.

#### Diffusion Loop 수정

```python
@torch.no_grad()
def generate(
    self,
    ...,
    attention_store: Optional[OviAttentionStore] = None,  # 새로 추가
):
    """
    Args:
        attention_store: Attention 저장을 위한 store 객체
    """

    # ... (초기화) ...

    # ========== Denoising Loop ==========
    for i, (t_v, t_a) in tqdm(enumerate(zip(timesteps_video, timesteps_audio))):

        # AttentionStore에 현재 timestep 설정
        if attention_store is not None:
            attention_store.set_timestep(int(t_v.item()))

        # ... (노이즈 예측) ...

        # Forward pass with attention storage
        pos_forward_args = {
            'vid_e': latents_video,
            'vid_seq_lens': video_seq_lens,
            'vid_grid_sizes': video_grid_sizes,
            'vid_freqs': video_freqs,
            'audio_e': latents_audio,
            'audio_seq_lens': audio_seq_lens,
            'audio_grid_sizes': audio_grid_sizes,
            'audio_freqs': audio_freqs,
            'context': text_emb,
            'context_lens': text_lens,
            'store_attention': (attention_store is not None)  # 플래그 전달
        }

        latents_video, latents_audio = self.model.forward(**pos_forward_args)

        # ... (노이즈 업데이트) ...

    # 마지막 timestep 저장
    if attention_store is not None:
        attention_store.finalize()

    return latents_video, latents_audio
```

**주요 변경사항**:
1. `attention_store` 파라미터 추가
2. 각 timestep마다 `set_timestep()` 호출
3. `store_attention` 플래그를 forward pass에 전달
4. Loop 종료 후 `finalize()` 호출

---

### 5. `inference.py`

메인 inference 스크립트에 attention 시각화 파이프라인을 통합했습니다.

#### 1) Config 로드 및 Flash Attention 설정

```python
# Config에서 설정 로드
visualize_attention = config.get("visualize_attention", False)
use_flash_attention = config.get("use_flash_attention", True)

# Flash Attention 토글
if not use_flash_attention:
    from ovi.modules.attention import set_flash_attention_enabled
    set_flash_attention_enabled(False)
    logger.info("Flash Attention: Disabled")
else:
    logger.info("Flash Attention: Enabled")
```

#### 2) AttentionStore 초기화

```python
attention_store = None
if visualize_attention:
    from ovi.utils.ovi_attention_viz import OviAttentionStore

    token_idx = config.get("visualize_token_idx", None)
    attention_store = OviAttentionStore(token_idx=token_idx)
    attention_store.enable()

    # 모든 video block에 attention_store 등록
    for block in ovi_engine.model.video_model.blocks:
        block.cross_attn.attention_store = attention_store

    logger.info(f"AttentionStore registered to {len(ovi_engine.model.video_model.blocks)} blocks")
```

#### 3) 비디오 생성 with Attention Storage

```python
# Generate video with attention tracking
generated_video, generated_audio = ovi_engine.generate(
    text_emb=text_emb,
    text_lens=text_lens,
    ...,
    attention_store=attention_store,  # AttentionStore 전달
)
```

#### 4) 메모리 최적화: 프레임 백업

```python
# 비디오 저장 전에 시각화할 프레임만 백업 (메모리 절약)
backup_frame = None
if visualize_attention and attention_store is not None:
    frame_idx = config.get("visualize_frame_idx", 0)
    if frame_idx < generated_video.shape[1]:
        backup_frame = generated_video[:, frame_idx, :, :].copy()
        logger.info(f"Backed up frame {frame_idx} for attention visualization")

# 비디오 저장
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{base_name}_{timestamp}.mp4"
save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
logger.info(f"Video saved: {output_path}")

# 메모리 해제 (중요!)
del generated_video
if generated_audio is not None:
    del generated_audio
import gc
gc.collect()
logger.info("Memory freed after video save")
```

**메모리 최적화 이유**:
- `generated_video`: [3, 121, 512, 992] = 약 180MB
- `save_video()`에서 moviepy가 모든 프레임을 list로 변환 → 메모리 2배
- GIF 생성에는 단 1개 프레임만 필요 → 미리 백업 후 삭제

#### 5) Attention GIF 생성

```python
if visualize_attention and attention_store is not None and backup_frame is not None:
    logger.info(f"Creating attention visualization with {len(attention_store.attention_maps)} timesteps...")

    # 토큰 정보 로깅
    tokens = ovi_engine.text_model.tokenizer.tokenizer.tokenize(text)
    logger.info("=" * 60)
    logger.info("Text Tokens and Indices:")
    for idx, token in enumerate(tokens):
        logger.info(f"  Index {idx:2d}: '{token}'")
    logger.info("=" * 60)

    # 백업된 프레임을 PIL Image로 변환
    frame = np.transpose(backup_frame, (1, 2, 0))  # [C, H, W] → [H, W, C]
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    frame_pil = Image.fromarray(frame)

    # 토큰 이름 추출
    token_idx = config.get("visualize_token_idx", None)
    if token_idx is not None and token_idx < len(tokens):
        token_name = tokens[token_idx]
    else:
        token_name = "all_avg"

    # Attention GIF 생성
    from ovi.utils.ovi_attention_viz import create_attention_video

    frame_idx = config.get("visualize_frame_idx", 0)
    attn_output_path = f"{base_name}_{timestamp}_attention_frame{frame_idx}_{token_name}.gif"

    create_attention_video(
        attention_store=attention_store,
        frame=frame_pil,
        output_path=attn_output_path,
        token_name=token_name,
        frame_idx=frame_idx
    )

    logger.info(f"Attention GIF saved: {attn_output_path}")
```

**주요 포인트**:
1. 토큰 인덱스와 실제 단어를 로그로 표시 (사용자 편의)
2. 백업된 프레임만 사용하여 메모리 절약
3. 파일명에 timestam + frame_idx + token_name 포함

---

## 핵심 기술

### 1. Flash Attention vs Standard Attention

| 특징 | Flash Attention | Standard Attention |
|------|----------------|-------------------|
| 속도 | 빠름 ⚡ | 느림 🐢 |
| 메모리 (VRAM) | 적음 💚 | 많음 🔴 |
| Attention Weight 저장 | ❌ 불가능 | ✅ 가능 |
| 시각화 | ❌ 불가능 | ✅ 가능 |

**구현 방식**:
```python
# Config 기반 토글
if USE_FLASH_ATTENTION:
    x = flash_attention_2_or_3(q, k, v)  # 빠르지만 weight 저장 안 됨
else:
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # weight 추출 가능
```

---

### 2. Cross Attention 구조

Ovi의 Text-to-Video Cross Attention:

```
Text Embeddings (K, V)     Video Latents (Q)
     [B, 512, C]     ×     [B, 61952, C]
                    ↓
            Attention Weights
              [B, H, Lq, Lk]
         [1, 24, 61952, 512]
                    ↓
        Spatial Map (특정 토큰)
                 [Lq]
               [61952]
                    ↓
            Reshape to 2D
         [F×H×W] → [H×W]
       [31×32×62] → [32×62]
```

- **B**: Batch size (1)
- **H**: Attention heads (24)
- **Lq**: Query sequence length (비디오 패치 수)
  - 31 frames × 32 height patches × 62 width patches = 61,952
- **Lk**: Key sequence length (텍스트 토큰 수, 예: 512)

---

### 3. Spatial Map 추출

특정 토큰의 spatial attention pattern을 추출하는 과정:

```python
# Input: attn_weights [B, H, Lq, Lk]
# 예: [1, 24, 61952, 512]

# 1. 특정 토큰 선택 (예: token_idx=1 → "man")
token_attn = attn_weights[0, :, :, token_idx]  # [H, Lq] = [24, 61952]

# 2. Multi-head 평균
spatial_map = token_attn.mean(dim=0)  # [Lq] = [61952]

# 3. GPU → CPU 즉시 전송 (VRAM 절약)
spatial_map_cpu = spatial_map.detach().cpu()

# 4. Running average 누적
if temp_sum is None:
    temp_sum = spatial_map_cpu
else:
    temp_sum += spatial_map_cpu
layer_count += 1

# 5. Timestep 완료 시 평균화
avg_spatial_map = temp_sum / layer_count  # [61952]
```

---

### 4. 2D Spatial Map 재구성

1차원 spatial map을 2D 이미지로 변환:

```python
# Input: spatial_map [Lq] = [61952]
# grid_sizes [B, 3] = [[31, 32, 62]]  (F, H, W)

num_frames = 31
h_patches = 32
w_patches = 62
frame_idx = 0

# 1. 해당 프레임의 패치만 추출
patches_per_frame = h_patches * w_patches  # 32 × 62 = 1984
start_idx = frame_idx * patches_per_frame  # 0
end_idx = start_idx + patches_per_frame    # 1984

frame_attn = spatial_map[start_idx:end_idx]  # [1984]

# 2. 2D로 reshape
attn_map_2d = frame_attn.reshape(h_patches, w_patches)  # [32, 62]

# 3. 원본 프레임 크기로 리사이즈
attn_map_resized = cv2.resize(attn_map_2d, (992, 512))  # [512, 992]

# 4. Normalize (0-1)
attn_map_norm = (attn_map_resized - attn_map_resized.min()) / \
                (attn_map_resized.max() - attn_map_resized.min() + 1e-8)

# 5. Colormap 적용 (빨강 = 높음, 파랑 = 낮음)
heatmap = cv2.applyColorMap(
    (attn_map_norm * 255).astype(np.uint8),
    cv2.COLORMAP_JET
)

# 6. 원본 프레임과 블렌딩
blended = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
```

---

## 사용 방법

### 1. Config 설정

`ovi/configs/inference/inference_fusion.yaml` 파일 수정:

```yaml
# Flash Attention 비활성화 (시각화를 위해 필수!)
use_flash_attention: false

# Attention 시각화 활성화
visualize_attention: true

# 시각화할 토큰 선택
visualize_token_idx: 1  # 0="A", 1="man", 2="running", 3="at", 4="the", 5="park"
                        # null = 전체 평균

# 시각화할 프레임 선택
visualize_frame_idx: 0  # 0 = 첫 번째 프레임

# 프롬프트 (토큰 인덱스는 이 프롬프트 기준)
text: "A man running at the park"
```

### 2. 실행

```bash
python3 inference.py --config-file ovi/configs/inference/inference_fusion.yaml
```

### 3. 출력 확인

생성된 파일:
```
outputs/
├── A_man_running_at_the_park_512x992_103_0_20251102_193244.mp4
└── A_man_running_at_the_park_512x992_103_0_20251102_193244_attention_frame0_▁man.gif
```

**GIF 내용**:
- 50개 프레임 (각 diffusion timestep의 attention)
- "man" 토큰이 어디에 attend하는지 히트맵으로 표시
- 빨간색 = 높은 attention, 파란색 = 낮은 attention

### 4. 다른 토큰 시각화

실행 시 콘솔에 토큰 인덱스가 자동으로 로그됩니다:

```
============================================================
Text Tokens and Indices:
  Index  0: '▁A'
  Index  1: '▁man'
  Index  2: '▁running'
  Index  3: '▁at'
  Index  4: '▁the'
  Index  5: '▁park'
============================================================
```

원하는 토큰의 인덱스를 확인하고 config를 수정:

```yaml
# "running" 시각화
visualize_token_idx: 2

# "park" 시각화
visualize_token_idx: 5

# 전체 평균
visualize_token_idx: null
```

### 5. 다른 프레임 시각화

```yaml
# 중간 프레임 시각화 (121프레임 중)
visualize_frame_idx: 60

# 마지막 프레임 시각화
visualize_frame_idx: 120
```

---

## 문제 해결

개발 과정에서 발생한 주요 에러와 해결 방법입니다.

### 1. Flash Attention Import 경고

**에러**:
```
UserWarning: flash_attention imported but unused
```

**원인**: `from .attention import flash_attention, attention` 후 `attention()`만 사용

**해결**: `flash_attention()` 함수 내부에서 조건부 분기하도록 수정
```python
def flash_attention(...):
    if not USE_FLASH_ATTENTION:
        return attention(...)  # Fallback
    # Flash Attention 구현
```

---

### 2. Tokenizer AttributeError

**에러**:
```python
AttributeError: 'HuggingfaceTokenizer' object has no attribute 'tokenize'
```

**원인**: Ovi의 tokenizer는 nested 구조
```python
ovi_engine.text_model.tokenizer  # HuggingfaceTokenizer (wrapper)
ovi_engine.text_model.tokenizer.tokenizer  # AutoTokenizer (실제 tokenizer)
```

**해결**: Nested access 사용
```python
# 잘못된 방법
tokens = ovi_engine.text_model.tokenizer.tokenize(text)

# 올바른 방법
tokens = ovi_engine.text_model.tokenizer.tokenizer.tokenize(text)
```

---

### 3. PIL Image 변환 에러

**에러**:
```python
TypeError: Cannot handle this data type: (1, 1, 992), <f4
```

**원인**: `generated_video`를 직접 iterate하면서 잘못된 shape 전달

**해결**: 올바른 indexing과 transpose
```python
# 잘못된 방법
for frame in generated_video:  # frame shape이 이상함
    pil_frame = Image.fromarray(frame)

# 올바른 방법
frame = generated_video[:, frame_idx, :, :]  # [3, H, W]
frame = np.transpose(frame, (1, 2, 0))       # [H, W, 3]
frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
pil_frame = Image.fromarray(frame)
```

---

### 4. RAM 부족 "Killed" 에러

**에러**:
```
Killed  (process terminated by OS)
```

**원인**: 20GB RAM 제한에서 메모리 부족
- Full attention [B,H,Lq,Lk] 저장 시도 → 4.5TB 필요
- 121프레임을 PIL로 변환 시 메모리 2배
- moviepy buffer도 메모리 소비

**해결책 1**: Immediate spatial map extraction
```python
# Before: 전체 저장 (4.5TB)
self.attention_maps[timestep] = attn_weights  # [B, H, Lq, Lk]

# After: 즉시 spatial map 추출 (12.4MB)
spatial_map = attn_weights[0, :, :, token_idx].mean(dim=0)  # [Lq]
spatial_map_cpu = spatial_map.detach().cpu()
```

**해결책 2**: Running average
```python
# Before: 모든 레이어 저장 (30 layers × 50 steps)
layer_attns = []
for layer in layers:
    layer_attns.append(attn_weights)

# After: Running average
if temp_sum is None:
    temp_sum = spatial_map
else:
    temp_sum += spatial_map
layer_count += 1
# Timestep 완료 시 평균화
avg = temp_sum / layer_count
```

**해결책 3**: 프레임 백업
```python
# Before: 전체 비디오 메모리에 유지
save_video(path, generated_video, ...)  # 180MB × 2 (moviepy buffer)
create_gif(generated_video, ...)         # 여전히 360MB 사용

# After: 필요한 프레임만 백업
backup_frame = generated_video[:, 0, :, :].copy()  # 1.5MB
save_video(path, generated_video, ...)
del generated_video  # 메모리 해제
gc.collect()
create_gif(backup_frame, ...)  # 1.5MB만 사용
```

---

### 5. FusionModel forward() 파라미터 에러

**에러**:
```python
TypeError: FusionModel.forward() got an unexpected keyword argument 'store_attention'
```

**원인**: Call chain의 일부 함수에만 `store_attention` 파라미터 추가

**해결**: 모든 함수에 파라미터 추가
```python
# 수정해야 할 함수들
FusionModel.forward(..., store_attention=False)
single_fusion_block_forward(..., store_attention=False)
single_fusion_cross_attention_ffn_forward(..., store_attention=False)
single_fusion_cross_attention_forward(..., store_attention=False)
```

---

### 6. grid_sizes Unpacking 에러

**에러**:
```python
TypeError: cannot unpack non-iterable int object
```

**원인**: `grid_sizes`를 `[B, F, 2]` 형식으로 잘못 가정
- 실제: `[B, 3]` where 3 = (F, H, W)

**해결**: 올바른 unpacking
```python
# 잘못된 방법
h_patches, w_patches = grid_sizes[0, frame_idx]  # frame_idx로 indexing 불가

# 올바른 방법
if len(grid_sizes.shape) == 2 and grid_sizes.shape[1] == 3:
    num_frames, h_patches, w_patches = grid_sizes[0].tolist()
    num_frames, h_patches, w_patches = int(num_frames), int(h_patches), int(w_patches)
```

---

### 7. BFloat16 NumPy 변환 에러

**에러**:
```python
TypeError: Got unsupported ScalarType BFloat16
```

**원인**: NumPy가 BFloat16을 직접 지원하지 않음

**해결**: Float32로 변환 후 NumPy 변환
```python
# 잘못된 방법
attn_map_2d = frame_attn.reshape(h_patches, w_patches).numpy()

# 올바른 방법
frame_attn = frame_attn.float()  # BFloat16 → Float32
attn_map_2d = frame_attn.reshape(h_patches, w_patches).numpy()
```

---

## 메모리 최적화

### 메모리 사용량 비교

| 방법 | VRAM | RAM | 총 메모리 | 채택 |
|------|------|-----|-----------|------|
| Full Attention 저장 | ~100GB | ~4.4TB | ~4.5TB | ❌ |
| Layer-averaged Attention | ~50GB | ~100GB | ~150GB | ❌ |
| **Running Avg + Spatial Map** | **~2GB** | **~620MB** | **~2.6GB** | ✅ |

### 최종 채택 전략

#### 1. Immediate Spatial Map Extraction

```python
def store(self, attn_weights, q_shape, k_shape, grid_sizes):
    """
    GPU에서 즉시 spatial map 추출
    [B, H, Lq, Lk] → [Lq]
    """
    # 1. 특정 토큰의 spatial map 추출 (GPU)
    if self.token_idx is not None:
        spatial_map = attn_weights[0, :, :, self.token_idx].mean(dim=0)  # [Lq]
    else:
        spatial_map = attn_weights[0].mean(dim=(0, 2))  # [Lq]

    # 2. 즉시 CPU로 전송 (VRAM 해제)
    spatial_map_cpu = spatial_map.detach().cpu()

    # 메모리 절약:
    # Before: [1, 24, 61952, 512] × 4 bytes = 3.1GB (BFloat16이면 1.5GB)
    # After:  [61952] × 4 bytes = 248KB
```

#### 2. Running Average Across Layers

```python
def store(self, spatial_map_cpu):
    """
    레이어별로 즉시 누적하여 평균화
    """
    # 누적
    if self.temp_spatial_sum is None:
        self.temp_spatial_sum = spatial_map_cpu
    else:
        self.temp_spatial_sum += spatial_map_cpu
    self.temp_layer_count += 1

    # 메모리 절약:
    # Before: [61952] × 30 layers = 7.4MB per timestep
    # After:  [61952] × 1 (running sum) = 248KB per timestep
```

#### 3. Timestep Finalization

```python
def set_timestep(self, t: int):
    """
    Timestep 전환 시 이전 timestep 평균화하여 저장
    """
    # 이전 timestep 평균 계산
    if self.current_timestep is not None and self.temp_spatial_sum is not None:
        avg_spatial_map = self.temp_spatial_sum / max(self.temp_layer_count, 1)
        self.attention_maps[self.current_timestep] = {
            'spatial_map': avg_spatial_map,  # [61952]
            'grid_sizes': self.temp_grid_sizes
        }

    # 리셋
    self.current_timestep = t
    self.temp_spatial_sum = None
    self.temp_layer_count = 0

    # 메모리:
    # 50 timesteps × 248KB = 12.4MB (전체 저장)
```

#### 4. Frame Backup Before Video Save

```python
# 비디오 저장 전
backup_frame = generated_video[:, frame_idx, :, :].copy()  # 1.5MB

# 비디오 저장 (메모리 2배 사용)
save_video(output_path, generated_video, ...)  # 180MB → 360MB

# 즉시 메모리 해제
del generated_video  # 180MB 해제
gc.collect()

# GIF 생성 (백업 프레임만 사용)
create_attention_video(..., frame=backup_frame, ...)  # 1.5MB
```

### 최종 메모리 프로파일

**VRAM (GPU)**:
- VAE: ~1.5GB
- Text Encoder (임시): ~0.5GB
- Diffusion Model (임시, CPU offload): ~0GB
- Spatial map extraction (임시): ~3GB
- **Total**: ~2GB

**RAM (CPU)**:
- Models (CPU offload): ~11GB (5B params × 2 bytes FP16)
- Generated video (임시): ~180MB
- Attention maps (50 timesteps): ~12.4MB
- Backup frame: ~1.5MB
- **Total**: ~12GB
- **Peak during save_video()**: ~12.5GB

**20GB RAM에서 안전하게 실행 가능!** ✅

---

## 참고사항

### Attend-and-Excite 논문과의 비교

**Attend-and-Excite**:
- Stable Diffusion (text-to-image)
- 16×16 cross attention maps 저장
- 각 timestep마다 attention 강화/억제

**Ovi Implementation**:
- Text-to-Video 생성
- 32×62 spatial maps (512×992 해상도)
- Attention 시각화 목적 (수정 없음)
- 50 timesteps × 31 frames의 시공간 정보

### Flash Attention 버전

Ovi는 Flash Attention 2와 3을 모두 지원합니다:

```python
try:
    from flash_attn import flash_attn_func
    FLASH_VERSION = 2
except ImportError:
    try:
        from flash_attn_interface import flash_attn_func
        FLASH_VERSION = 3
    except ImportError:
        FLASH_VERSION = None
```

시각화 시에는 자동으로 PyTorch standard attention으로 fallback됩니다.

### 패치 기반 구조

Ovi는 비디오를 패치로 나누어 처리합니다:

- **입력**: 512×992 비디오, 31프레임
- **패치 크기**: 16×16 pixels
- **패치 수**:
  - Height: 512 ÷ 16 = 32 patches
  - Width: 992 ÷ 16 = 62 patches
  - Frames: 31
  - **Total**: 31 × 32 × 62 = 61,952 patches

Attention map의 각 값은 하나의 패치(16×16 영역)에 대한 attention 강도를 나타냅니다.

---

## 추가 개선 사항 (향후)

1. **Multi-token 시각화**: 여러 토큰을 동시에 다른 색으로 표시
2. **Temporal attention**: 프레임 간 attention 시각화
3. **Interactive viewer**: 웹 기반 인터랙티브 시각화 도구
4. **Attention editing**: Attend-and-Excite처럼 attention 조작 기능
5. **Comparison mode**: 여러 프롬프트의 attention 비교

---

## 라이선스

본 attention 시각화 기능은 Ovi 프로젝트의 라이선스를 따릅니다.

---

## 문의

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해주세요.

---

**마지막 업데이트**: 2025-11-02
