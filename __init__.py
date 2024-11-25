from .clip_combiner import CLIPSplitter, CLIPCombiner

NODE_CLASS_MAPPINGS = {
    "CLIPSplitter": CLIPSplitter,
    "CLIPCombiner": CLIPCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPSplitter": "CLIP Splitter",
    "CLIPCombiner": "CLIP Combiner",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 