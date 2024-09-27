import os

WORKSPACE = "/data/discover-08/liwy/workspace/HRDA"

def run_evaluation(ckpt_path: str):
    os.chdir(WORKSPACE)
    
    CONFIGs = {
        "cs": "configs/hrda/gtaHR2csHR_hrda_r101.py",
        "idd": "configs/hrda/gtaHR2iddHR_hrda_r101.py",
    }
    
    SAVE_ROOT = os.path.join(WORKSPACE, "auto-evaluate/results")
    ckpt_name = os.path.basename(ckpt_path).split(".")[0]
    
    CMDs = [
        f"bash test_c7w.sh {CONFIGs['cs']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-cs.log')}",
        f"bash test_c7w.sh {CONFIGs['idd']} {ckpt_path} {os.path.join(SAVE_ROOT, f'{ckpt_name}-idd.log')}",
    ]
    
    for cmd in CMDs:
        os.system(cmd)


def main_loop():
    os.chdir(WORKSPACE)
    WAITING_FOR_EVALUATION = os.path.join(WORKSPACE, "auto-evaluate/pretrained")
    EVALUATED = os.path.join(WORKSPACE, "auto-evaluate/evaluated")
    
    # for each ckpt in WAITING_FOR_EVALUATION
    while os.listdir(WAITING_FOR_EVALUATION):
        for ckpt in os.listdir(WAITING_FOR_EVALUATION):
            ckpt_path = os.path.join(WAITING_FOR_EVALUATION, ckpt)
            run_evaluation(ckpt_path)
            os.rename(ckpt_path, os.path.join(EVALUATED, ckpt))



if __name__ == "__main__":
    main_loop()  # enter main loop
