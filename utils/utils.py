import yaml
import json
import argparse
import importlib
import omegaconf.dictconfig

from .register import Registers


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(config):
    conf_dict = {}
    for key, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            conf_dict[key] = namespace2dict(value)
        else:
            conf_dict[key] = value
    return conf_dict


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_runner(runner_name, config):
    runner = Registers.runners[runner_name](config)
    return runner

def save_to_file(save_content, file_path, save_format='yaml'):
    save_formats = ['txt', 'json', 'yaml']
    assert save_format in save_formats, "save format should be txt, yaml, or json"

    with open(file_path, 'w') as f:
        if save_format == 'txt':
            f.write(save_content)
        elif save_format == 'yaml':
            yaml.dump(save_content, f)
        else:
            f.write(json.dumps(save_content, indent=4))