import os

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

# @torch.inference_mode()
@torch.amp.autocast('cuda')

class Tracker:
    def __init__(self, minor_edge_size = 480, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cutie = get_default_model(device)
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = minor_edge_size

    def track(self, frame_stack, init_mask, starting_frame = 0):
        init_mask = init_mask.astype(np.uint8)
        objects = np.unique(init_mask)
        objects = objects[objects != 0].tolist()
        mask_stack = np.zeros((frame_stack.shape[0], frame_stack.shape[1], frame_stack.shape[2]), dtype=np.uint8)
        mask_stack[starting_frame] = init_mask
        init_mask = torch.from_numpy(init_mask).to(self.device)
        frame_stack = [to_tensor(frame).to(self.device).float() for frame in frame_stack]
        for i in range(len(frame_stack)):
            frame_stack[i].requires_grad = False
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # Track forward
            if starting_frame < len(frame_stack) - 1:
                _ = self.processor.step(frame_stack[starting_frame], init_mask, objects=objects)
                for i in range(starting_frame + 1, len(frame_stack)):
                    output_prob = self.processor.step(frame_stack[i])
                    mask = self.processor.output_prob_to_mask(output_prob)
                    mask_stack[i] = mask.cpu().numpy()
            # Track backward
            if starting_frame > 0:
                _ = self.processor.step(frame_stack[starting_frame], init_mask, objects=objects)
                for i in range(starting_frame - 1, -1, -1):
                    output_prob = self.processor.step(frame_stack[i])
                    mask = self.processor.output_prob_to_mask(output_prob)
                    mask_stack[i] = mask.cpu().numpy()
        return mask_stack

if __name__ == "__main__":
    tracker = Tracker()
                
        
