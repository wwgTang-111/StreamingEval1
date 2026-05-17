import argparse
import torch
import gc
import numpy as np
import time
import os
import json
from typing import Optional, Tuple, List
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoConfig
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from torch.multiprocessing import Process, Queue, Manager, Event
from decord import VideoReader
import queue
from transformers import StoppingCriteria, StoppingCriteriaList
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
class FirstTokenStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.first_token_time = None
    def __call__(self, input_ids, scores, **kwargs):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        return False
def get_all_from_queue(q, count, video_id="Unknown"):
    items = []
    for _ in range(count):
        try:
            item = q.get(timeout=10)
            items.append(item)
        except queue.Empty:
            print(f"warning")
            break
    return items
def back_prompt(prompt, num_frames, patches_per_frame, vision_start_token, image_pad_token):
    vision_end_token = "?"  
    frame_tokens = (
        vision_start_token +                    
        image_pad_token * patches_per_frame +  
        vision_end_token                       
    )
    all_frames = frame_tokens * num_frames
    full_prompt = (
        "?" +
        all_frames +
        prompt +
        "?"
    )
    return full_prompt
def resolve_vision_special_tokens(tokenizer, config) -> Tuple[Optional[int], int, Optional[int], Optional[int]]:
    image_token_id = getattr(config, "?", None)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("?")
        if image_token_id == tokenizer.unk_token_id:
            image_token_id = None
    if image_token_id is None:
        raise ValueError("BUG")
    vision_start_id = tokenizer.convert_tokens_to_ids("?")
    if vision_start_id == tokenizer.unk_token_id:
        vision_start_id = None
    vision_end_id = tokenizer.convert_tokens_to_ids("?")
    if vision_end_id == tokenizer.unk_token_id:
        vision_end_id = None
    print(f"token ID: {image_token_id} (?)")
    if vision_start_id and vision_end_id:
        print(f"token IDs: start={vision_start_id}, end={vision_end_id}")
    return image_token_id, vision_start_id, vision_end_id
def build_prompt(task, question, options, _anno_, index):
##Construct according to the dataset
    return task
def video_stream_similator(len_queue, anno, frame_queue, log_queue, video_end_queue, args1, data_ready_event, frame_processed_event, video_fps=1.0,
                           play_speed=1.0, memory_bank_len=14):
   
    task_types = ["a", "b", "c"]
   
    total_videos = 0
    for task_type in task_types:
        total_videos += len(anno.get(task_type, []))
    pbar = tqdm(total=total_videos, desc="Processing Videos", unit="video")
    for task_type in task_types:
        current_task_list = anno.get(task_type, [])
        if not current_task_list:
            continue
        for i in range(0, len(current_task_list)):
            _anno_ = current_task_list[i]
            video_id_base = _anno_["id"]
            task_name = _anno_["task"]
            sub_tasks = []
            if task_type == "1":
                test_info = _anno_["test_info"]
                for sub_i in range(len(test_info)):
                    chunk_path = os.path.join(args1.chunked_dir, f"{video_id_base}_{sub_i}.mp4")
                    sub_prompt = build_prompt(task_name, question=None, options=None, _anno_=_anno_, index=sub_i)
                    sub_tasks.append((chunk_path, sub_prompt, "", f"{video_id_base}_{sub_i}"))
            else:
                chunk_path = os.path.join(args1.chunked_dir, f"{video_id_base}.mp4")
                question = _anno_.get("question", "")
                sub_prompt = build_prompt(task_name, question, options=_anno_.get("options", []), _anno_=_anno_, index=None)
                sub_tasks.append((chunk_path, sub_prompt, question, video_id_base))

            for chunk_video_path, prompt, question, video_id in sub_tasks:
                if not os.path.exists(chunk_video_path):
                    print(f"pass")
                    pbar.update(1)
                    continue
                try:
                    vr = VideoReader(chunk_video_path)
                    sample_fps = max(1, round(vr.get_avg_fps() / video_fps))
                    all_frame_idx = [j for j in range(0, len(vr), sample_fps)]
                    video = vr.get_batch(all_frame_idx).asnumpy()
                    length = video.shape[0]
                    for start in range(0, length):
                        end = min(start + 1, length)
                        video_clip = video[start:end]
                        frame_queue.put(video_clip)
                        if not frame_processed_event.wait(timeout=60):
                            print(f"warning")
                        frame_processed_event.clear()
                    time.sleep(10)
                    signal = (prompt, question, task_name, task_type, video_id, length)
                    video_end_queue.put(signal)
                    data_ready_event.wait()
                    data_ready_event.clear()
                    pbar.update(1)
                    
                except Exception as e:
                    print(f'Simulator Exception: {e}')
                    pbar.update(1)
                finally:
                    if 'vr' in locals() and vr is not None:
                        del vr
                    if 'video' in locals() and video is not None:
                        del video
    
    pbar.close()
  
    video_end_queue.put(None)

def frame_memory_manager(memory_bank, processor, frame_queue, log_queue, model_dir, model_ready_event, frame_processed_event, memory_bank_len, save_subdir_name, resize_resolution=None):
    
    base_save_dir = ""
    os.makedirs(base_save_dir, exist_ok=True)
    time_save_path = os.path.join(base_save_dir, "")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}
    )
    vision_model = full_model.model.visual
    vision_model = vision_model.to("cuda:0", dtype=torch.bfloat16)
    vision_model.eval()
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    model_ready_event.set()
    image_processor = processor.image_processor
    memory_size = memory_bank_len
    frame_cnt = 0 
    while True:
        try:
            video_clip = frame_queue.get()
            start_time = time.perf_counter()
            frame_cnt += 1
            from PIL import Image
            frames_list = []
            #data pre-processing

            pixel_values = image_inputs["pixel_values"]
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

            with torch.inference_mode():
                image_embedding = vision_model(pixel_values, grid_thw=grid_thw)
            if memory_bank.qsize() >= memory_size:
                try:
                    memory_bank.get_nowait()
                except queue.Empty:
                    pass
            image_embedding_cpu = image_embedding.detach().to("cpu", dtype=torch.float16)
            cpu_embedding = image_embedding_cpu.numpy()
            memory_bank.put(cpu_embedding)
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            with open(time_save_path, "a", encoding="utf-8") as f:
                f.write(f"{processing_time:.6f}\n")
            del frames_list, image_inputs, pixel_values, image_embedding, image_embedding_cpu, cpu_embedding, video_clip
            frame_processed_event.set()
        except Exception as e:
            print(f'MemManager Exception: {e}')
            frame_processed_event.set() 
            time.sleep(100)
def run_inference(inference_model, processor, tokenizer, text_prompt, video_embeddings_list, 
                  vision_start_token, image_pad_token, image_token_id, save_path, save_time=False):
    
    num_frames = len(video_embeddings_list)
    patches_num = sum(emb.shape[0] for emb in video_embeddings_list)
    vision_end_token = "?"
    all_frames_str = ""
    for emb in video_embeddings_list:
        current_patches = emb.shape[0]
        frame_str = (
            vision_start_token + 
            image_pad_token * current_patches + 
            vision_end_token
        )
        all_frames_str += frame_str
    prompt = "?"
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(inference_model.device)
    input_ids = inputs["input_ids"][0]
    video_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(video_positions) != patches_num:
        print(f"warning!")
    input_embedding_layer = inference_model.get_input_embeddings()
    inputs_embed = input_embedding_layer(input_ids)
    torch_video_embeddings = [
        torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
        for emb in video_embeddings_list
    ]
    all_video_patches = torch.cat(torch_video_embeddings, dim=0)
    final_embeds_parts = []
    last_pos = 0
    for idx, pos in enumerate(video_positions):
        if pos > last_pos:
            text_part = inputs_embed[last_pos:pos]
            final_embeds_parts.append(text_part)
        if idx >= len(all_video_patches):
            break
        patch_emb = all_video_patches[idx:idx+1].to(inputs_embed.device, dtype=inputs_embed.dtype)
        final_embeds_parts.append(patch_emb)
        last_pos = pos + 1
    final_text = inputs_embed[last_pos:]
    final_embeds_parts.append(final_text)
    final_inputs_embeds = torch.cat(final_embeds_parts, dim=0).unsqueeze(0)
    first_token_timer = FirstTokenStoppingCriteria()
    stopping = StoppingCriteriaList([first_token_timer])
    expanded_mask = torch.ones(final_inputs_embeds.shape[:2], dtype=torch.long, device=final_inputs_embeds.device)
    if isinstance(tokenizer.eos_token_id, list):
        eos_token_id = tokenizer.eos_token_id[0] if tokenizer.eos_token_id else None
    else:
        eos_token_id = tokenizer.eos_token_id
    llm_start_time = time.perf_counter()
    generated_ids = inference_model.generate(
        inputs_embeds=final_inputs_embeds,
        #Check for demand filling
    )
    llm_end_time = time.perf_counter()
    response_time = llm_end_time - llm_start_time
    if first_token_timer.first_token_time is not None:
        ttft = first_token_timer.first_token_time - llm_start_time
    else:
        ttft = 0.0
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"results: {output_text}")
    if save_time:
        with open(save_path, "a", encoding="utf-8") as u:
            u.write(f"{response_time:.6f} | {output_text}\n")
    else:
        with open(save_path, "a", encoding="utf-8") as u:
            u.write(output_text + "\n")     
    with open("answer.txt", "a", encoding="utf-8") as f:
        f.write(output_text + "\n")
    return ttft, output_text

def run_experiment(args, target_tokens, save_subdir_name, resize_resolution=None):
    
    torch.multiprocessing.set_start_method('spawn', force=True)
    len_queue = Queue()
    log_queue = Queue()
    video_end_queue = Queue()
    frame_queue = Queue(maxsize=3000)
    data_ready_event = Event()
    model_ready_event = Event()
    frame_processed_event = Event() 
    processes = []
    
    with open("question.json", "r") as f:
        annotations = json.load(f)
    backward_anno = []
    realtime_anno = []
    forward_anno = []
    backward_tasks = ["1", "2", "3"]
    realtime_tasks = ["4", "5", "6"]
    forward_tasks = ["7", "8", "9"]
    all_tasks = backward_tasks + realtime_tasks + forward_tasks
    if args.task == ["ALL"]:
        target_tasks = all_tasks
        print("all is over")
    else:
        target_tasks = []
        for t in args.task:
            if t == "a":
                target_tasks.extend(forward_tasks)
            elif t == "b":
                target_tasks.extend(backward_tasks)
            elif t == "c":
                target_tasks.extend(realtime_tasks)
            else:
                target_tasks.append(t)
        print(f"js {target_tasks}")

    for anno in annotations:
        if anno["task"] in target_tasks:
            if anno["task"] in a:
                backward_anno.append(anno)
            if anno["task"] in b:
                realtime_anno.append(anno)
            if anno["task"] in c:
                forward_anno.append(anno)

    anno = {
        "p": p11,
        "q": p21,
        "d": p31
    }

    model_dir = "?"

    inference_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}
    )
    inference_model.eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else tokenizer.eos_token_id[0]
    image_token_id, vision_start_id, vision_end_id = resolve_vision_special_tokens(tokenizer, config)
    vision_start_token = tokenizer.convert_ids_to_tokens(vision_start_id) 
    image_pad_token = tokenizer.convert_ids_to_tokens(image_token_id) 
    with Manager():
        memory_bank = Queue()
        p3 = Process(target=frame_memory_manager,
                     args=(memory_bank, processor, frame_queue, log_queue, model_dir, model_ready_event, frame_processed_event,  save_subdir_name, resize_resolution))
        processes.append(p3)
        p3.start()
        if not model_ready_event.wait(timeout=1800):
            print(f"quit")
            p3.terminate()
            return
        args1 = args
        p2 = Process(target=video_stream_similator,
                     args=(len_queue, anno, frame_queue, log_queue, video_end_queue, args1, data_ready_event, frame_processed_event, args1.video_fps, args1.play_speed))
        processes.append(p2)
        p2.start()
        while True:
            try:
                signal = video_end_queue.get()
                if signal is None:
                    print("quit")
                    break
                if signal is not None:
                    prompt_text = signal[0]
                    question_text = signal[1]
                    task_name = signal[2]
                    task_type = signal[3]
                    video_id = signal[4]
                    frame_count = signal[5]
                    if frame_count == 0:
                        print(f"pass")
                        data_ready_event.set()
                        continue
                    video_embeddings_list = []
                    wait_start = time.perf_counter()
                    max_wait_s = 60
                    while not video_embeddings_list and (time.perf_counter() - wait_start) < max_wait_s:
                        video_embeddings_list = get_all_from_queue(memory_bank, frame_count, video_id)
                        if not video_embeddings_list:
                            time.sleep(0.5)
                    if not video_embeddings_list:
                        print(f"pass")
                        data_ready_event.set()
                        continue
                    torch_video_embeddings = [
                        torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb
                        for emb in video_embeddings_list
                    ]
                    patches_per_frame = torch_video_embeddings[0].shape[0]
                    all_video_patches = torch.cat(torch_video_embeddings, dim=0) 
                    if all_video_patches.shape[0] > target_tokens:
                        all_video_patches = all_video_patches[-target_tokens:] 
                    current_total_tokens = all_video_patches.shape[0]
                    final_video_embeddings_list = []
                    start_idx = 0   
                    #joint
                    if remaining_tokens > 0:
                        partial_frame_emb = all_video_patches[start_idx : start_idx + remaining_tokens]
                        final_video_embeddings_list.append(partial_frame_emb)
                    save_dir = ""  
                    os.makedirs(save_dir, exist_ok=True)
                    save_path_1 = os.path.join(save_dir, f"{task_name}.txt")
                    ttft, _ = run_inference(
                        inference_model, processor, tokenizer, prompt_text, final_video_embeddings_list,
                        vision_start_token, image_pad_token, image_token_id, save_path_1, save_time=False
                    )
                    with open(os.path.join(f"", ""), "a", encoding="utf-8") as f:
                        f.write(f"{target:.6f}\n")
                    del video_embeddings_list, torch_video_embeddings, all_video_patches, final_video_embeddings_list
                    data_ready_event.set()
                    continue
            except Exception as e:
                import traceback
                print("=" * 80)
                print(f"wrong: {str(e)}")
                traceback.print_exc()
                print("=" * 80)
    for p in processes:
        p.terminate()
    del inference_model, processor, tokenizer, config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"over")

def main(args):
    memory_configs = [
        {"q": "s", "w": "u", "e": None},
    ]
    resolution_configs = []
    all_configs = memory_configs + resolution_configs
    print(f"coming soon")
    for config in all_configs:
        try:
            run_experiment(args, config["q"], config["w"], config["e"])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(10)
        except Exception as e:
            print(f" failure")
            import traceback
            traceback.print_exc()
    print("all is over")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--chunked_dir", type=str, default="",)
    parser.add_argument("--task", nargs='+', default=["ALL"])
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--play_speed", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
