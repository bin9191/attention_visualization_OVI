import os
import sys
import logging
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt
from ovi.utils.utils import get_arguments
from ovi.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.ovi_attention_viz import OviAttentionStore, create_attention_video



def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def main(config, args): 

    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)
    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must be less than or equal to world_size and world_size must be divisible by sp_size."

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size)
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."
        ## TODO: assert not sharding t5 etc...


    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")
    
    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    # validate inputs before loading model to not waste time if input is not valid
    text_prompt = config.get("text_prompt")
    image_path = config.get("image_path", None)
    assert config.get("mode") in ["t2v", "i2v", "t2i2v"], f"Invalid mode {config.get('mode')}, must be one of ['t2v', 'i2v', 't2i2v']"
    text_prompts, image_paths = validate_and_process_user_prompt(text_prompt, image_path, mode=config.get("mode"))
    if config.get("mode") != "i2v":
        logging.info(f"mode: {config.get('mode')}, setting all image_paths to None")
        image_paths = [None] * len(text_prompts)
    else:
        assert all(p is not None and os.path.isfile(p) for p in image_paths), f"In i2v mode, all image paths must be provided.{image_paths}"

    logging.info("Loading OVI Fusion Engine...")
    ovi_engine = OviFusionEngine(config=config, device=device, target_dtype=target_dtype)
    logging.info("OVI Fusion Engine loaded!")

    # Attention Visualization 설정
    visualize_attention = config.get("visualize_attention", False)
    attention_store = None

    if visualize_attention:
        if config.get("use_flash_attention", True):
            logging.warning("visualize_attention is True but use_flash_attention is also True!")
            logging.warning("Setting use_flash_attention to False for visualization...")
            config.use_flash_attention = False
            # Flash attention 설정 다시 적용
            from ovi.modules.attention import set_flash_attention_enabled
            set_flash_attention_enabled(False)

        logging.info("Initializing AttentionStore for visualization...")
        # token_idx를 전달하여 메모리 절약
        token_idx = config.get("visualize_token_idx", None)
        attention_store = OviAttentionStore(token_idx=token_idx)
        attention_store.enable()

        # 모든 cross_attn 레이어에 attention_store 등록
        for block in ovi_engine.model.video_model.blocks:
            block.cross_attn.attention_store = attention_store

        logging.info(f"AttentionStore registered to {len(ovi_engine.model.video_model.blocks)} blocks")

    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV data
    all_eval_data = list(zip(text_prompts, image_paths))

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        # No SP: treat each GPU as its own group
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Data distribution - by SP groups
    total_files = len(all_eval_data)

    require_sample_padding = False
    
    if total_files == 0:
        logging.error(f"ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        # Pad to match number of SP groups
        remainder = total_files % num_sp_groups
        if require_sample_padding and remainder != 0:
            pad_count = num_sp_groups - remainder
            all_eval_data += [all_eval_data[0]] * pad_count
        
        # Distribute across SP groups
        this_rank_eval_data = all_eval_data[sp_group_id :: num_sp_groups]

    for _, (text_prompt, image_path) in tqdm(enumerate(this_rank_eval_data)):
        video_frame_height_width = config.get("video_frame_height_width", None)
        seed = config.get("seed", 100)
        solver_name = config.get("solver_name", "unipc")
        sample_steps = config.get("sample_steps", 50)
        shift = config.get("shift", 5.0)
        video_guidance_scale = config.get("video_guidance_scale", 4.0)
        audio_guidance_scale = config.get("audio_guidance_scale", 3.0)
        slg_layer = config.get("slg_layer", 11)
        video_negative_prompt = config.get("video_negative_prompt", "")
        audio_negative_prompt = config.get("audio_negative_prompt", "")
        for idx in range(config.get("each_example_n_times", 1)):
            # Attention store 초기화 (시각화 모드인 경우)
            if visualize_attention and attention_store is not None:
                attention_store.clear()

            generated_video, generated_audio, generated_image = ovi_engine.generate(text_prompt=text_prompt,
                                                                    image_path=image_path,
                                                                    video_frame_height_width=video_frame_height_width,
                                                                    seed=seed+idx,
                                                                    solver_name=solver_name,
                                                                    sample_steps=sample_steps,
                                                                    shift=shift,
                                                                    video_guidance_scale=video_guidance_scale,
                                                                    audio_guidance_scale=audio_guidance_scale,
                                                                    slg_layer=slg_layer,
                                                                    video_negative_prompt=video_negative_prompt,
                                                                    audio_negative_prompt=audio_negative_prompt,
                                                                    attention_store=attention_store)  # 추가!

            if sp_rank == 0:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = format_prompt_for_filename(text_prompt)
                output_path = os.path.join(output_dir, f"{formatted_prompt}_{'x'.join(map(str, video_frame_height_width))}_{seed+idx}_{global_rank}_{timestamp}.mp4")
                # 비디오 정보 로깅
                num_frames = generated_video.shape[1]  # (C, F, H, W)
                logging.info(f"Generated video shape: {generated_video.shape} ({num_frames} frames)")

                # 메모리 절약: Attention 시각화용 프레임 백업 (비디오 저장 전)
                backup_frame = None
                if visualize_attention and attention_store is not None:
                    frame_idx = config.get("visualize_frame_idx", 0)
                    if frame_idx < generated_video.shape[1]:
                        backup_frame = generated_video[:, frame_idx, :, :].copy()  # (C, H, W)

                save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
                if generated_image is not None:
                    generated_image.save(output_path.replace('.mp4', '.png'))

                # 비디오 메모리 해제
                del generated_video
                if generated_audio is not None:
                    del generated_audio
                import gc
                gc.collect()
                logging.info("Memory freed after video save")

                # Attention 시각화 저장 (시각화 모드인 경우)
                if visualize_attention and attention_store is not None:
                    timesteps = attention_store.get_all_timesteps()
                    if timesteps:
                        logging.info(f"Creating attention visualization with {len(timesteps)} timesteps...")

                        # 텍스트 토큰화
                        # HuggingfaceTokenizer의 내부 AutoTokenizer에 접근
                        tokens = ovi_engine.text_model.tokenizer.tokenizer.tokenize(text_prompt)

                        # 토큰 인덱스 정보 출력
                        logging.info("=" * 60)
                        logging.info("Text Tokens and Indices:")
                        for i, token in enumerate(tokens):
                            logging.info(f"  Index {i:2d}: '{token}'")
                        logging.info("=" * 60)

                        # Attention 비디오 생성
                        token_idx = config.get("visualize_token_idx", None)
                        frame_idx = config.get("visualize_frame_idx", 0)

                        # 백업한 프레임 사용
                        from PIL import Image
                        import numpy as np
                        if backup_frame is not None:
                            frame = np.transpose(backup_frame, (1, 2, 0))  # (H, W, C)
                            # [0, 255] 범위로 변환
                            if frame.max() <= 1.0:
                                frame = np.clip(frame, 0, 1)
                                frame = (frame * 255).astype(np.uint8)
                            else:
                                frame = frame.astype(np.uint8)
                            target_frame = Image.fromarray(frame)
                            video_frames = [target_frame]
                        else:
                            logging.error(f"backup_frame is None")
                            video_frames = []

                        if token_idx is not None and token_idx < len(tokens):
                            token_name = tokens[token_idx]
                            attn_output_path = output_path.replace('.mp4', f'_attention_frame{frame_idx}_{token_name}.gif')
                        else:
                            attn_output_path = output_path.replace('.mp4', f'_attention_frame{frame_idx}_all.gif')

                        create_attention_video(
                            attention_store=attention_store,
                            video_frames=video_frames,
                            text_tokens=tokens,
                            token_idx=token_idx,
                            output_path=attn_output_path,
                            fps=10,
                            frame_idx=frame_idx
                        )
                        logging.info(f"Visualizing attention for frame {frame_idx}/{len(video_frames)-1}")
                        logging.info(f"Attention visualization saved to {attn_output_path}")
                    else:
                        logging.warning("No attention maps captured during generation!")
        


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config,args=args)