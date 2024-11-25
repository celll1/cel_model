import comfy.sd
import folder_paths
import torch
import copy
import comfy.model_management

class CLIPSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "clip": ("CLIP",),
        }}
    
    RETURN_TYPES = ("CLIP_L", "CLIP_G", "CLIP_T5")
    RETURN_NAMES = ("CLIP_L", "CLIP_G", "T5")
    FUNCTION = "split_clip"
    CATEGORY = "cel_model/clip"

    def split_clip(self, clip):
        # CLIPの種類を判定
        if hasattr(clip.cond_stage_model, 'clip_l') and hasattr(clip.cond_stage_model, 'clip_g'):
            if hasattr(clip.cond_stage_model, 't5xxl'):  # SD3の場合
                splits = [
                    self._create_single_clip(clip, 'clip_l'),
                    self._create_single_clip(clip, 'clip_g'),
                    self._create_single_clip(clip, 't5xxl')
                ]
            else:  # SDXLの場合
                splits = [
                    self._create_single_clip(clip, 'clip_l'),
                    self._create_single_clip(clip, 'clip_g'),
                    None
                ]
        else:
            # 単一のCLIPモデルの場合
            splits = [clip, None, None]

        # 返したCLIPの種類をログに出力
        for i, split in enumerate(splits):
            if split is not None:
                print(f"Returned CLIP {i+1}: {type(split.cond_stage_model)}")

        return tuple(splits)

    def _create_single_clip(self, clip, model_type):
        if getattr(clip.cond_stage_model, model_type, None) is None:
            return None
        new_clip = clip.clone()
        new_clip.cond_stage_model = getattr(clip.cond_stage_model, model_type)
        return new_clip

class CLIPCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_l": ("CLIP_L",),
            "model_type": (["sdxl", "sd3"], {"default": "sdxl"}),
        },
        "optional": {
            "clip_g": ("CLIP_G",),
            "clip_t5": ("CLIP_T5",),
        }}
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "combine_clip"
    CATEGORY = "cel_model/clip"

    def combine_clip(self, clip_l, model_type, clip_g=None, clip_t5=None):
        device = "cpu"  # 初期化時はCPUで
        
        try:
            # エンベディングディレクトリの取得
            embedding_directory = None
            if hasattr(clip_l.tokenizer, 'clip_l'):
                embedding_directory = clip_l.tokenizer.clip_l.embedding_directory
            elif hasattr(clip_l.tokenizer, 'l'):
                embedding_directory = clip_l.tokenizer.l.embedding_directory
            
            if model_type == "sdxl":
                if clip_g is None:
                    raise ValueError("SDXL requires both CLIP_L and CLIP_G")
                from comfy.sdxl_clip import SDXLClipModel, SDXLTokenizer
                combined_model = SDXLClipModel(device=device)
                combined_model.clip_l = clip_l.cond_stage_model
                combined_model.clip_g = clip_g.cond_stage_model
                
                # SDXLのトークナイザー設定
                new_clip = clip_l.clone()
                new_clip.cond_stage_model = combined_model
                new_clip.tokenizer = SDXLTokenizer(embedding_directory=embedding_directory)
            
            elif model_type == "sd3":
                from comfy.text_encoders.sd3_clip import SD3ClipModel, SD3Tokenizer
                # メモリ管理のために一時的にモデルをアンロード
                comfy.model_management.unload_all_models()
                
                has_t5 = clip_t5 is not None
                combined_model = SD3ClipModel(
                    clip_l=True,
                    clip_g=clip_g is not None,
                    t5=has_t5,
                    device=device
                )
                combined_model.clip_l = clip_l.cond_stage_model
                if clip_g is not None:
                    combined_model.clip_g = clip_g.cond_stage_model
                if has_t5:
                    combined_model.t5xxl = clip_t5.cond_stage_model

                # SD3用の新しいCLIPオブジェクトを作成
                new_clip = clip_l.clone()
                new_clip.cond_stage_model = combined_model
                new_clip.tokenizer = SD3Tokenizer(embedding_directory=embedding_directory)

            # 必要に応じてデバイスに移動
            target_device = comfy.model_management.intermediate_device()
            if target_device != "cpu":
                new_clip.cond_stage_model.to(target_device)
            
            return (new_clip,)
            
        except Exception as e:
            comfy.model_management.unload_all_models()
            raise e