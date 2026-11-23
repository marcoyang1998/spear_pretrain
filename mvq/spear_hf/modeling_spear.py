# modeling_spear.py

from transformers import PreTrainedModel
from .configuration_spear import SpearConfig
from .spear_model import SpearModel as model


class SpearModel(PreTrainedModel):
    config_class = SpearConfig

    def __init__(self, config: SpearConfig):
        super().__init__(config)
        self.model = model(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_audio(self, audio_path):
        return self.model.load_audio(audio_path)