import yacs.config as config
import copy

def merge_dict(_default_dict, ref_dict):
    if type(_default_dict) is dict:
        default_dict = copy.deepcopy(_default_dict)
    else:
        default_dict = _default_dict
    for key in ref_dict:
        try:
            if type(default_dict) is dict:
                if key in default_dict:
                    if type(default_dict[key]) is dict:
                        default_dict[key] = merge_dict(default_dict[key],
                                                       ref_dict[key])
                        default_dict[key] = ref_dict[key]
                else:
                    default_dict[key] = ref_dict[key]
            else:
                default_dict[key] = ref_dict[key]
        except Exception as err:
            print(err)
            print('default_dict', default_dict)
            print('ref_dict', ref_dict)
            raise err
    return default_dict

def make_config(config_file):
    with open(config_file, 'r') as f:
        cfg = config.load_cfg(f)
    return cfg.clone()
    
# https://stackoverflow.com/questions/1305532/convert-nested-python-dict-to-object/1305663
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


if "__main__" in __name__:
    print(make_config('../../exps/01/exp.yaml'))