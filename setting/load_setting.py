import json

CFG_PATH = 'setting.json'
with open(CFG_PATH, "r") as f:
    CFG = json.load(f)

MODEL_CFG_PREFIX = CFG['FILE_CFG']['MODEL_CFG_PREFIX']
DATASET_SUFFIX = CFG['FILE_CFG']['DATASET_SUFFIX']
CKPT_FILE_PREFIX = CFG['FILE_CFG']['CKPT_FILE_PREFIX']
ATTACKER_CFG = CFG['ATTACKER_CFG']
REMAIN_LIST = CFG['EXP_CFG']['REMAIN_LIST']

def validate_check(model_name, dataset_name, attacker_name):
    models = MODEL_CFG_PREFIX.keys()
    datasets = DATASET_SUFFIX.keys()
    attackers = ATTACKER_CFG[dataset_name].keys()

    assert model_name in models and dataset_name in datasets and attacker_name in attackers, \
        f'`model_name`, `dataset_name`, `attacker_name` expected in {models}, {datasets}, {attackers}, but got {model_name}, {dataset_name}, {attacker_name}.'
    
    assert model_name in ATTACKER_CFG[dataset_name][attacker_name].keys(), \
        f'Attacker `{attacker_name}` has no implement for model `{model_name}`.'

def get_attacker_params(model_name, dataset_name, attacker_name):
    validate_check(model_name, dataset_name, attacker_name)

    attacker_params = ATTACKER_CFG[dataset_name][attacker_name][model_name]
    model_config_path = MODEL_CFG_PREFIX[model_name] + DATASET_SUFFIX[dataset_name] + '.py'
    model_ckpt_path = CKPT_FILE_PREFIX[model_name] + DATASET_SUFFIX[dataset_name] + '.pth'
    
    attacker_params.update({'cfg_file': model_config_path, 'ckpt_file': model_ckpt_path})

    return attacker_params

def get_remain_list(attacker_name):
    return REMAIN_LIST[attacker_name]