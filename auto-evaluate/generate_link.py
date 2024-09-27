PATHS = '''acdc_r101_seed1_iter40000="work_dirs/local-basic/231116_1745_gta2acdc_hrda_r101_seed1_1a8c6/iter_40000.pth"
darkzurich_r101_seed1_iter40000="work_dirs/local-basic/231116_1758_gta2darkzurich_hrda_r101_seed1_9066f/iter_40000.pth"
cs_r101_seed1_iter40000="work_dirs/local-basic/230920_1706_gtaHR2csHR_hrda_r101_seed1_21de0/latest.pth"
idd_r101_seed1_iter40000="work_dirs/local-basic/230920_1715_gtaHR2iddHR_hrda_r101_seed1_477b9/latest.pth"
'''

# Goto
# ln -s work_dirs/local-basic/231030_2350_gtaHR2csHR_hrda_r50_seed1_dfed1/iter_10000.pth auto-evaluate/pretrained/cs_r50_seed1_iter10000.pth
# ...
def generate_ln_commands(paths_string):
    lines = paths_string.split("\n")
    commands = []

    for line in lines:
        if len(line) == 0:
            continue
        # Extract the key and path from each line
        key, path = line.split('=')
        path = path.strip('"')
        
        # get the absolute path
        import os
        path = os.path.join("/data/discover-08/liwy/workspace/HRDA/", path)
        
        # Generate the ln command
        command = f'ln -s {path} auto-evaluate/pretrained/{key}.pth'
        commands.append(command)

    return '\n'.join(commands)

print(generate_ln_commands(PATHS))
