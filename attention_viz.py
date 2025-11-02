import abc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from diffusers import StableDiffusionPipeline
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

# %%
# 1. AttentionControl 및 수정된 TimestepAttentionStore 정의
# (기반: utils/ptp_utils.py)

class AttentionControl(abc.ABC):
    """
    AttentionControl의 기본 클래스 (ptp_utils.py에서 가져옴)
    """
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_step > self.stop_index:
            self.reset()
            return attn
        
        # 'forward' 메소드를 호출하여 TimestepAttentionStore의 로직 실행
        attn = self.forward(attn, is_cross, place_in_unet)
        
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, stop_index: int = 1000, num_att_layers: int = 16):
        self.cur_step = 0
        self.num_att_layers = num_att_layers
        self.cur_att_layer = 0
        self.stop_index = stop_index


class TimestepAttentionStore(AttentionControl):
    """
    [★★수정된 AttentionStore★★]
    기존 AttentionStore는 모든 타임스텝의 맵을 평균냈지만,
    이 클래스는 타임스텝(t)을 key로 사용하여 맵을 분리 저장합니다.
    """
    @staticmethod
    def get_empty_store():
        # {해상도: [맵 리스트]} 형태의 딕셔너리
        return defaultdict(list)

    def __init__(self, stop_index: int = 1000, num_att_layers: int = 16):
        super().__init__(stop_index, num_att_layers)
        # {타임스텝: {해상도: [맵 리스트]}}
        self.step_store = defaultdict(self.get_empty_store)
        self.self_attn_store = defaultdict(self.get_empty_store)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # self.cur_t는 AttentionControl의 부모 클래스(CrossAttnProcessor)에서
        # U-Net의 forward pass 시 설정됩니다.
        t = self.cur_t 
        
        # 해상도 (h*w)
        h = attn.shape[0] // self.batch_size
        
        if is_cross:
            # 현재 타임스텝(t)의 해당 해상도(h)에 어텐션 맵 저장
            self.step_store[t][h].append(attn)
        else:
            self.self_attn_store[t][h].append(attn)
        return attn

    def __init__(self, batch_size=1, stop_index: int = 1000, num_att_layers: int = 16):
        super().__init__(stop_index, num_att_layers)
        self.batch_size = batch_size
        # {타임스텝: {해상도: [맵 리스트]}}
        self.step_store = defaultdict(self.get_empty_store)
        self.self_attn_store = defaultdict(self.get_empty_store)


# %%
# 2. (수정된) U-Net에 Attention Store 등록 함수
# (기반: utils/ptp_utils.py)

def register_attention_control(model: StableDiffusionPipeline, controller: AttentionControl):
    """
    U-Net의 모든 Cross-Attention 레이어에 controller를 등록(설치)합니다.
    """
    
    # diffusers 0.11.0 이상 버전 호환
    def ca_forward(self, place_in_unet: str):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **cross_attention_kwargs,
        ):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            
            # [★★핵심★★]
            # 어텐션 가중치(probs)를 controller로 전달
            # controller는 이 값을 TimestepAttentionStore.forward()로 처리
            controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states
        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            # CrossAttention 레이어를 찾으면 forward 함수를 교체
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net_name, net in sub_nets:
        if "down" in net_name:
            cross_att_count += register_recr(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_recr(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_recr(net, 0, "up")
    
    # num_att_layers를 controller에 설정
    controller.num_att_layers = cross_att_count
    print(f"Total {cross_att_count} CrossAttention layers registered.")


# %%
# 3. (수정된) 타임스텝별 어텐션 집계 함수
# (기반: utils/ptp_utils.py - aggregate_attention)

def aggregate_attention_for_timestep(
    timestep_store: Dict[int, List[torch.Tensor]], # TimestepAttentionStore.step_store[t]
    res: int,
    from_where: List[str], # ('up', 'down', 'mid')
    select: int = 0
) -> torch.Tensor:
    """
    특정 타임스텝(t)의 저장소(timestep_store)에서 맵을 집계합니다.
    """
    out = []
    attention_maps = timestep_store
    
    # 1. 해상도(res)에 맞춰 맵 선택
    num_pixels = res ** 2
    for h, att_list in attention_maps.items():
        if h == num_pixels:
            out.extend(att_list)
    
    if len(out) == 0:
        # 이 타임스텝에는 해당 해상도의 맵이 없음 (초기 단계)
        return None

    # 2. 맵 평균내기
    out = torch.stack(out, dim=0)
    out = out.sum(0) / out.shape[0] # (batch_size * num_heads, seq_len, num_tokens)
    
    # 3. 토큰 선택 (select=0은 uncond/CFG 제외)
    # diffusers 파이프라인은 (uncond, cond) 2개를 1 배치로 처리
    # batch_size=1일 때, [0]은 uncond, [1]은 cond
    # Attend-and-Excite는 0번(uncond)을 사용 (pipeline_... 265라인)
    # 하지만 시각화는 보통 1번(cond)을 봄. 여기서는 1로 가정.
    # 만약 원본 repo와 동일하게 하려면 select=0
    select = 1 # 0: uncond, 1: text prompt
    out = out[select::2] # text-conditional 맵만 선택
    
    return out


# %%
# 4. 시각화 헬퍼 함수
# (가져오기: utils/vis_utils.py, utils/gaussian_smoothing.py)

class GaussianSmoothing(nn.Module):
    """
    utils/gaussian_smoothing.py에서 가져옴
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * np.sqrt(2 * np.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.conv = F.conv2d if dim == 2 else F.conv3d

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    utils/vis_utils.py (show_cam_on_image)
    히트맵(mask)을 원본 이미지(img)에 오버레이합니다.
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
        img_float = np.float32(img) / 255
        cam = heatmap + img_float
    
    # 3. [0, 1] 범위를 [0, 255]로 복구
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def show_image_relevance(image_relevance: torch.Tensor,
                         img: Image.Image,
                         low_res: int = 16,
                         smooth_attentions: bool = True,
                         sigma: float = 0.5,
                         kernel_size: int = 3) -> np.ndarray:
    """
    utils/vis_utils.py (show_image_relevance)
    어텐션 맵(image_relevance)을 받아 최종 히트맵 이미지로 변환합니다.
    """
    img_h, img_w = img.size
    
    # 1. 2D 맵으로 변환 및 크기 조정
    image_relevance = image_relevance.reshape(1, 1, low_res, low_res)
    image_relevance = F.interpolate(image_relevance, size=(img_h, img_w), mode='bilinear')
    image_relevance = image_relevance.squeeze() # (img_h, img_w)
    
    # 2. CPU 및 NumPy로 이동
    image_relevance = image_relevance.cpu().numpy()

    # 3. 스무딩 (optional)
    if smooth_attentions:
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2)
        smoothing = smoothing.to(image_relevance.device)
        input_tensor = torch.from_numpy(image_relevance).unsqueeze(0).unsqueeze(0)
        input_tensor = F.pad(input_tensor, (1, 1, 1, 1), mode='reflect')
        image_relevance = smoothing(input_tensor).squeeze(0).squeeze(0).numpy()

    # 4. 정규화 (0~1)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    
    # 5. 히트맵 생성 및 오버레이
    vis = show_cam_on_image(np.array(img), image_relevance)
    vis = Image.fromarray(vis)
    return vis

def aggregate_attention_all_timesteps(
    controller: TimestepAttentionStore,
    res: int,
    from_where: List[str] = ("up", "down", "mid")
) -> torch.Tensor:
    """
    [신규 함수]
    컨트롤러에 저장된 '모든' 타임스텝의 맵을 평균내어
    '최종' 집계 맵을 반환합니다.
    (기반: utils/ptp_utils.py의 aggregate_attention 로직)
    """
    all_timestep_maps = []
    captured_timesteps = controller.step_store.keys()
    
    if not captured_timesteps:
        return None

    for t in captured_timesteps:
        timestep_store = controller.step_store[t]
        
        # 'aggregate_attention_for_timestep' 함수를 재사용하여 
        # 해당 타임스텝(t)의 맵을 먼저 집계합니다.
        aggregated_map_t = aggregate_attention_for_timestep(
            timestep_store=timestep_store,
            res=res,
            from_where=from_where
        )
        
        if aggregated_map_t is not None:
            all_timestep_maps.append(aggregated_map_t)
    
    if len(all_timestep_maps) == 0:
        return None
    
    # (Time, Heads, Pixels, Tokens) 차원으로 스택
    all_timestep_maps = torch.stack(all_timestep_maps, dim=0)
    
    # 시간(Time) 차원에 대해 평균을 내어 최종 맵 생성
    # (Heads, Pixels, Tokens)
    final_map = all_timestep_maps.mean(dim=0)
    
    return final_map

# %%
# 5. 메인 실행 로직: 파이프라인 실행 및 비디오 생성

if __name__ == "__main__":
    
    # --- 1. 설정 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # 시각화할 프롬프트 및 토큰 인덱스
    PROMPT = "a cat and a dog"
    # 토큰 인덱스 (Tokenizer 기준)
    # <start> a  cat and a  dog <end>
    #   0     1   2   3   4   5    6
    TOKEN_INDEX_TO_VISUALIZE = 5  # "dog"
    TOKEN_NAME = "dog" # 파일명 저장용
    
    LOW_RES_MAP_SIZE = 16 # U-Net 중간 해상도 (16x16, 32x32, 64x64...)
    NUM_INFERENCE_STEPS = 50
    SEED = 42
    
    print("Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    
    
    # --- 2. AttentionStore 등록 ---
    print("Registering attention controller...")
    # (batch_size=1, num_att_layers=16 -> 16은 기본값이며 register_attention_control에서 덮어씀)
    controller = TimestepAttentionStore(batch_size=1) 
    register_attention_control(pipe, controller)

    
    # --- 3. 파이프라인 1회 실행 (Attention Map 저장) ---
    print("Running pipeline to capture attention maps...")
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    # 파이프라인을 실행하면, register된 controller가 호출되며
    # controller.step_store에 모든 타임스텝의 맵이 자동으로 저장됨.
    with torch.no_grad():
        generated_image = pipe(
            prompt=PROMPT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            output_type="pil"
        ).images[0]
        
    generated_image.save("generated_image.png")
    print(f"Generated image saved to 'generated_image.png'")


    # --- 4. 타임스텝별 시각화 및 비디오 프레임 생성 ---
    print("Generating frames for attention video...")
    frames = []
    
    # 저장된 타임스텝 (e.g., [981, 961, ..., 1])
    captured_timesteps = sorted(controller.step_store.keys(), reverse=True)
    
    for t in captured_timesteps:
        # 1. 현재 타임스텝(t)의 맵 저장소 가져오기
        timestep_store = controller.step_store[t]
        
        # 2. 해당 타임스텝의 맵 집계 (LOW_RES_MAP_SIZE x LOW_RES_MAP_SIZE)
        aggregated_map = aggregate_attention_for_timestep(
            timestep_store=timestep_store,
            res=LOW_RES_MAP_SIZE,
            from_where=("up", "down", "mid")
        )
        
        if aggregated_map is None:
            # 이 타임스텝/해상도에는 맵이 없음 (스킵)
            continue
            
        # 3. 특정 토큰의 맵 추출
        # aggregated_map.shape: (num_heads, seq_len, num_tokens)
        # 평균내기: (seq_len, num_tokens)
        avg_map = aggregated_map.mean(dim=0) # (77, 77) -> (pixels, tokens)
        
        # 맵 크기(16x16=256)와 토큰(77)
        # ptp_utils.aggregate_attention은 (heads, 256, 77)을 반환
        # avg_map = aggregated_map.mean(dim=0) # (256, 77)
        
        # [수정] aggregate_attention_for_timestep 반환 shape: (heads, pixels, tokens)
        avg_map = aggregated_map.mean(dim=0) # (pixels, tokens)
        
        # 4. 특정 토큰(TOKEN_INDEX_TO_VISUALIZE)의 맵 선택
        token_map = avg_map[:, TOKEN_INDEX_TO_VISUALIZE] # (pixels,)
        
        # 5. 히트맵 이미지 생성
        frame_vis = show_image_relevance(
            image_relevance=token_map,
            img=generated_image,
            low_res=LOW_RES_MAP_SIZE
        )
        
        frames.append(frame_vis)

    if not frames:
        print(f"No frames generated. Check if LOW_RES_MAP_SIZE ({LOW_RES_MAP_SIZE}) is correct.")
    else:
        # --- 5. 비디오(GIF) 저장 ---
        output_filename = f"attention_video_{TOKEN_NAME}.gif"
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            duration=150,  # 프레임 간 딜레이 (ms)
            loop=0  # 0: 무한 반복
        )
        print(f"Successfully saved attention video to '{output_filename}'")

    # --- 6. [신규 추가] 최종 토큰별 맵 생성 (요청 1) ---
    print("\nGenerating final aggregated maps for all tokens...")

    # (A)에서 추가한 함수를 호출하여 모든 타임스텝의 맵을 집계
    final_aggregated_map = aggregate_attention_all_timesteps(
        controller=controller,
        res=LOW_RES_MAP_SIZE,
        from_where=("up", "down", "mid")
    )
    
    if final_aggregated_map is not None:
        # (Heads, Pixels, Tokens) -> (Pixels, Tokens)
        # 헤드(Head) 차원에 대해 평균
        final_avg_map = final_aggregated_map.mean(dim=0) 
        
        # 1. 프롬프트 토큰화 (인덱스 및 이름 추출)
        tokenizer = pipe.tokenizer
        text_inputs = tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        token_ids = text_inputs.input_ids[0]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        prompt_tokens = []
        token_indices = []
        
        # <start>, <end>, <pad> 토큰 제외
        for i, token in enumerate(tokens):
            if token_ids[i] == tokenizer.bos_token_id or \
               token_ids[i] == tokenizer.eos_token_id or \
               token_ids[i] == tokenizer.pad_token_id:
                continue
            prompt_tokens.append(token)
            token_indices.append(i)

        print(f"Generating maps for tokens: {prompt_tokens}")

        # 2. 토큰별(Token-wise) 맵 생성
        for token, index in zip(prompt_tokens, token_indices):
            # 이 토큰의 맵 추출 (Pixels,)
            token_map = final_avg_map[:, index] 
            
            # 히트맵 이미지 생성
            token_vis = show_image_relevance(
                image_relevance=token_map,
                img=generated_image,
                low_res=LOW_RES_MAP_SIZE
            )
            
            # 파일명 정제
            safe_token_name = token.replace('</w>', '').replace('<s>', '').replace('</s>', '').replace('.', '').replace('/', '')
            output_filename = f"final_attention_map_idx{index}_{safe_token_name}.png"
            token_vis.save(output_filename)
            print(f"Saved final map for token '{token}' to '{output_filename}'")

        # 3. "전체 토큰(All Tokens)" 맵 생성 (Max 집계)
        # 모든 '단어' 토큰들의 어텐션 맵을 모음 (Pixels, Num_Word_Tokens)
        all_word_maps = final_avg_map[:, token_indices]
        
        # 픽셀별로 가장 어텐션이 강한 토큰의 값을 선택 (Max aggregation)
        all_token_map, _ = torch.max(all_word_maps, dim=1) # (Pixels,)
        
        all_token_vis = show_image_relevance(
            image_relevance=all_token_map,
            img=generated_image,
            low_res=LOW_RES_MAP_SIZE
        )
        output_filename = "final_attention_map_ALL_TOKENS_MAX.png"
        all_token_vis.save(output_filename)
        print(f"Saved final max-aggregated map to '{output_filename}'")
        
    else:
        print("Could not generate final aggregated maps.")