import yaml
import os


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base_config, override_config):
    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    def __init__(self, config_path=None, **kwargs):
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)

        for key, value in kwargs.items():
            self._set_nested(key, value)

    def _set_nested(self, key, value):
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def get(self, key, default=None):
        keys = key.split('.')
        d = self.config
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key):
        return key in self.config


if __name__ == "__main__":
    config = load_config("configs/dense_lf.yaml")
    print(config)
