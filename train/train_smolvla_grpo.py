from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import torch.nn as nn
import json

policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")

# cfg = json.load(open(hf_hub_download("HuggingFaceVLA/smolvla_libero", "config.json")))
# print(cfg)

print(isinstance(policy, nn.Module))