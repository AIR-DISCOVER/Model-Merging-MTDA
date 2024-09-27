import os

WORKSPACE = "/data/discover-08/liwy/workspace/HRDA"
f = open(os.path.join(WORKSPACE, "auto-evaluate/generate_cmd.sh"), "w")

def run_evaluation(ckpt_path: str):
    os.chdir(WORKSPACE)
    
    CONFIGs = {
        "cs_r101": "configs/hrda/gtaHR2csHR_hrda_r101.py",
        "idd_r101": "configs/hrda/gtaHR2iddHR_hrda_r101.py",
        "acdc_r101": "configs/hrda/gtaHR2acdcHR_hrda_r101.py",
        "darkzurich_r101": "configs/hrda/gtaHR2darkzurichHR_hrda_r101.py",
        "cs_mitb5": "configs/hrda/gtaHR2csHR_hrda.py",
        "idd_mitb5": "configs/hrda/gtaHR2iddHR_hrda.py",
        "acdc_mitb5": "configs/hrda/gtaHR2acdcHR_hrda.py",
        "darkzurich_mitb5": "configs/hrda/gtaHR2darkzurichHR_hrda.py",
    }
    
    CONFIGS_ABS = {
        k: os.path.join(WORKSPACE, v) for k, v in CONFIGs.items()
    }
    
    SCRIPT_PATH = os.path.join(WORKSPACE, "test_c7w.sh")
    
    SAVE_ROOT = os.path.join(WORKSPACE, "auto-evaluate/results")
    ckpt_name = ".".join(os.path.basename(ckpt_path).split(".")[:-1])
    
    CMDs_r101 = [
        f"bash {SCRIPT_PATH} {CONFIGS_ABS['cs_r101']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-cs.log')}",
        f"bash {SCRIPT_PATH} {CONFIGS_ABS['idd_r101']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-idd.log')}",
        f"bash {SCRIPT_PATH} {CONFIGS_ABS['acdc_r101']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-acdc.log')}",
        f"bash {SCRIPT_PATH} {CONFIGS_ABS['darkzurich_r101']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-darkzurich.log')}",
    ]
    
    CMDs_mitb5 = [
        f"bash {SCRIPT_PATH} {CONFIGS_ABS['cs_mitb5']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-cs.log')}",
        f"bash {SCRIPT_PATH} {CONFIGS_ABS['idd_mitb5']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-idd.log')}",
        f"bash {SCRIPT_PATH} {CONFIGS_ABS['acdc_mitb5']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-acdc.log')}",
        f"bash {SCRIPT_PATH} {CONFIGS_ABS['darkzurich_mitb5']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-darkzurich.log')}",
    ]
    
    if "101" in ckpt_path:
        CMDs = CMDs_r101
    elif "b5" in ckpt_path:
        CMDs = CMDs_mitb5
    else:
        raise ValueError("Unknown ckpt_path")
    for cmd in CMDs:
        print(cmd, file=f)


def main_loop():
    os.chdir(WORKSPACE)
    WAITING_FOR_EVALUATION = os.path.join(WORKSPACE, "auto-evaluate/pretrained")
    EVALUATED = os.path.join(WORKSPACE, "auto-evaluate/evaluated")
    
    # for each ckpt in WAITING_FOR_EVALUATION
    # while os.listdir(WAITING_FOR_EVALUATION):
    for ckpt in os.listdir(WAITING_FOR_EVALUATION):
        ckpt_path = os.path.join(WAITING_FOR_EVALUATION, ckpt)
        run_evaluation(ckpt_path)
        # os.rename(ckpt_path, os.path.join(EVALUATED, ckpt))



if __name__ == "__main__":
    main_loop()  # enter main loop
