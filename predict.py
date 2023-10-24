# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import imageio
from omegaconf import open_dict
from hydra import compose, initialize
import cv2
import torch
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.args_utils import get_dataset_cfg
from gui.interactive_utils import (
    image_to_torch,
    torch_prob_to_numpy_mask,
    index_numpy_to_one_hot_torch,
    overlay_davis,
)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        with torch.inference_mode():
            initialize(
                version_base="1.3.2", config_path="cutie/config", job_name="eval_config"
            )
            cfg = compose(config_name="eval_config")
            _ = get_dataset_cfg(cfg)

            with open_dict(cfg):
                cfg["weights"] = "./weights/cutie-base-mega.pth"

            # Load the network weights
            cutie = CUTIE(cfg).cuda().eval()
            model_weights = torch.load(cfg.weights)
            cutie.load_weights(model_weights)
            self.processor = InferenceCore(cutie, cfg=cfg)

    def predict(
        self,
        video: Path = Input(description="Input video"),
        mask: Path = Input(description="Mask for the first frame"),
        max_frames: int = Input(
            description="Number of frames to process. Leave blank to process the entire video.",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        mask = np.array(Image.open(str(mask)))
        print(f"Masks detected: {np.unique(mask)}")
        num_objects = len(np.unique(mask)) - 1

        device = "cuda"
        torch.cuda.empty_cache()

        cap = cv2.VideoCapture(str(video))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        current_frame_index = 0

        image_list = []

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                while cap.isOpened():
                    # load frame-by-frame
                    _, frame = cap.read()
                    if frame is None or (
                        max_frames is not None and current_frame_index > max_frames
                    ):
                        break

                    # convert numpy array to pytorch tensor format
                    frame_torch = image_to_torch(frame, device=device)
                    if current_frame_index == 0:
                        # initialize with the mask
                        mask_torch = index_numpy_to_one_hot_torch(
                            mask, num_objects + 1
                        ).to(device)
                        # the background mask is not fed into the model
                        prediction = self.processor.step(
                            frame_torch, mask_torch[1:], idx_mask=False
                        )
                    else:
                        # propagate only
                        prediction = self.processor.step(frame_torch)

                    # argmax, convert to numpy
                    prediction = torch_prob_to_numpy_mask(prediction)
                    visualization = overlay_davis(frame, prediction)
                    image_list.append(visualization)

                    current_frame_index += 1

        output_video = "/tmp/output.mp4"
        writer = imageio.get_writer(output_video, format="FFMPEG", fps=frame_rate)
        for frame in image_list:
            writer.append_data(frame)
        writer.close()
        return Path(output_video)
