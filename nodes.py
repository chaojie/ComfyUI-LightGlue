from PIL import Image
import numpy as np
import json
import torch

DEVICE = ["cuda","cpu"]

class LightGlueLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (DEVICE, {"default":"cuda"}),
            }
        }
        
    RETURN_TYPES = ("SuperPoint","LightGlue",)
    RETURN_NAMES = ("extractor","matcher",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "LightGlue"

    def load_checkpoint(self,device):
        from .lightglue import LightGlue, SuperPoint

        # SuperPoint+LightGlue
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
        matcher = LightGlue(features='superpoint').eval().to(device)  # load the matcher

        return (extractor,matcher,)

class LightGlueSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "extractor": ("SuperPoint",),
                "matcher": ("LightGlue",),
                "image0": ("IMAGE",),
                "image1": ("IMAGE",),
                "device": (DEVICE, {"default":"cuda"}),
            }
        }

    RETURN_TYPES = ("STRING",{},{},{},)
    RETURN_NAMES = ("motionbrush","matches","points0","points1",)
    FUNCTION = "run_inference"
    CATEGORY = "LightGlue"

    def run_inference(self,extractor,matcher,image0,image1,device):
        from .lightglue.utils import load_image,load_pilimage, rbd

        image0 = 255.0 * image0[0].cpu().numpy()
        image0 = Image.fromarray(np.clip(image0, 0, 255).astype(np.uint8))

        image1 = 255.0 * image1[0].cpu().numpy()
        image1 = Image.fromarray(np.clip(image1, 0, 255).astype(np.uint8))

        image0=load_pilimage(image0).to(device)
        image1=load_pilimage(image1).to(device)
        
        feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
        feats1 = extractor.extract(image1)
        
        #image0 = load_image("/home/admin/ComfyUI/input/1.png")
        #image1 = load_image("/home/admin/ComfyUI/input/2.png")
        # match the features
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        #print(f'matches{matches}')
        #print(f'points0{points0}')
        #print(f'points1{points1}')
        trajs=torch.stack([points0,points1],1)

        return (json.dumps(trajs.tolist()),matches,points0,points1,)


NODE_CLASS_MAPPINGS = {
    "LightGlue Loader":LightGlueLoader,
    "LightGlue Simple":LightGlueSimple,
}

