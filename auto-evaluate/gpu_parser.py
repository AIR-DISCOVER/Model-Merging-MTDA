import subprocess

def parse_nvidia_smi_results():
    
    CMD = "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv"
    
    smi_output = subprocess.check_output(CMD, shell=True).decode("utf-8")
    
    """name, memory.total [MiB], memory.free [MiB]
NVIDIA GeForce RTX 3090, 24576 MiB, 11279 MiB
NVIDIA GeForce RTX 3090, 24576 MiB, 3057 MiB
NVIDIA GeForce RTX 3090, 24576 MiB, 2591 MiB
NVIDIA GeForce RTX 3090, 24576 MiB, 2889 MiB
NVIDIA GeForce RTX 3090, 24576 MiB, 3787 MiB
NVIDIA GeForce RTX 3090, 24576 MiB, 1473 MiB
NVIDIA GeForce RTX 3090, 24576 MiB, 9431 MiB
NVIDIA GeForce RTX 3090, 24576 MiB, 21409 MiB
    """
    
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


print(parse_nvidia_smi_results())
