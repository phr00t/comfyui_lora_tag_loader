from pathlib import Path
from nodes import LoraLoader
import folder_paths
import re

# Import ComfyUI files
import comfy.sd
import comfy.utils

class LoraTagLoader:
    def __init__(self):
        self.tag_pattern = "\<[0-9a-zA-Z\:\_\-\.\s\/\(\)\\\\]+\>"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "text": ("STRING", {"multiline": True}),
                              "normalize_weight":   ("FLOAT", {"default":  0, "min": 0, "max": 100.0, "step": 0.1, "round": 0.001}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, text, normalize_weight):
        # print(f"\nLoraTagLoader input text: { text }")

        founds = re.findall(self.tag_pattern, text)
        # print(f"\nfoound lora tags: { founds }")

        if len(founds) < 1:
            return (model, clip, text)

        model_lora = model
        clip_lora = clip
        
        loras = []
        wModels = []
        wClips = []
        
        max_clip = 0.0
        max_weight = 0.0
        
        lora_files = folder_paths.get_filename_list("loras")
        for f in founds:
            tag = f[1:-1]
            pak = tag.split(":")
            type = pak[0]
            if type != 'lora':
                continue
            name = None
            if len(pak) > 1 and len(pak[1]) > 0:
                name = pak[1]
            else:
                continue
            wModel = wClip = 0
            try:
                if len(pak) > 2 and len(pak[2]) > 0:
                    wModel = float(pak[2])
                    wClip = wModel
                if len(pak) > 3 and len(pak[3]) > 0:
                    wClip = float(pak[3])
            except ValueError:
                continue
            if name == None:
                continue
            lora_name = None
            for lora_file in lora_files:
                if Path(lora_file).name.startswith(name) or lora_file.startswith(name):
                    lora_name = lora_file
                    break
            if lora_name == None:
                print(f"bypassed lora tag: { (type, name, wModel, wClip) } >> { lora_name }")
                continue
            print(f"detected lora tag: { (type, name, wModel, wClip) } >> { lora_name }")
            
            if wClip > 0 or wModel > 0:
                max_clip += abs(wClip)
                max_weight += abs(wModel)
                
                loras.append(lora_name)
                wModels.append(wModel)
                wClips.append(wClip)
            
        for idx, l in enumerate(loras):
            if normalize_weight > 0:
                weight_scale = normalize_weight / max_weight if max_weight > 0 else 1.0
                clip_scale = normalize_weight / max_clip if max_clip > 0 else 1.0
            else:
                weight_scale = 1.0
                clip_scale = 1.0
            model_lora, clip_lora = LoraLoader().load_lora(model_lora, clip_lora, l, wModels[idx] * weight_scale, wClips[idx] * clip_scale)

        plain_prompt = re.sub(self.tag_pattern, "", text)
        return (model_lora, clip_lora, plain_prompt)

NODE_CLASS_MAPPINGS = {
    "LoraTagLoader": LoraTagLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "LoraTagLoader": "Load LoRA Tag",
}
