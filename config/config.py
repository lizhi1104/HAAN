import os
from pathlib import Path

import tomli


class ConfigManager:
    @classmethod
    def get_configs(cls, dataset):
        if dataset == "FineAction":
            return cls._get_fine_action_configs()
        elif dataset == "FineGym":
            return cls._get_fine_gym_configs()
        else:
            raise Exception("Invalid dataset!")

    @classmethod
    def _get_fine_action_configs(cls):
        config_file = os.path.join(Path(__file__).parent, "fine_action.toml")
        return cls._read_configs(config_file)

    @classmethod
    def _get_fine_gym_configs(cls):
        config_file = os.path.join(Path(__file__).parent, "fine_gym.toml")
        return cls._read_configs(config_file)

    @classmethod
    def _read_configs(cls, file):
        with open(file, "rb") as f:
            config = tomli.load(f)
        return config
