import os
import subprocess
import multiprocessing
from typing import List
from time import sleep

SLEEP_TIME = 180
memory_threshold = 22862

WORKSPACE = "/data/discover-08/liwy/workspace/HRDA"

def parse_nvidia_smi_results():
    CMD = "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv"
    smi_output = subprocess.check_output(CMD, shell=True).decode("utf-8")
    
    results = []
    for line in smi_output.strip().split("\n")[1:]:
        name, total, free = line.split(",")
        name = name.strip()
        total = int(total.strip().split(" ")[0])
        free = int(free.strip().split(" ")[0])
        
        results.append({
            "idx": len(results),
            "name": name,
            "total": total,
            "free": free,
        })
            
    return results

def run_subprocess_command(cmd):
    return subprocess.check_output(cmd, shell=True)

# return a 
def run_cmd_with_env(cmd: str, env: dict):
    env_str = " ".join([f"{k}={v}" for k, v in env.items()])
    cmd = f"{env_str} {cmd}"
    # return a promise of multiprocessing.Process
    return multiprocessing.Process(target=run_subprocess_command, args=(cmd, ))


def main_loop(cmd_to_execute: List[str]):
    next_cmd_idx = 0
    harvest_process_list = []
    
    while True:
        current_gpu_state = parse_nvidia_smi_results()
        left_gpu = [gpu for gpu in current_gpu_state if gpu["free"] > memory_threshold]
        
        if len(left_gpu) == 0:
            print("No GPU available, waiting...")
        
        while next_cmd_idx < len(cmd_to_execute) and len(left_gpu) > 0:
            cmd = cmd_to_execute[next_cmd_idx]
            gpu = left_gpu.pop(0)
            next_cmd_idx += 1
            
            print(f"Running {cmd} on GPU {gpu['idx']}")
            p = run_cmd_with_env(cmd, env={"CUDA_VISIBLE_DEVICES": str(gpu["idx"])})
            p.start()
            harvest_process_list.append(p)

        if next_cmd_idx >= len(cmd_to_execute):
            break
        
        sleep(SLEEP_TIME)
    
    for p in harvest_process_list:
        p.join()


if __name__ == "__main__":
    cmd_to_execute = open(os.path.join(WORKSPACE, "auto-evaluate/generate_cmd.sh"), "r").read().split("\n")
    main_loop(cmd_to_execute)

