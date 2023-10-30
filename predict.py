# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import Optional
import os
import shutil
import imageio
import tempfile
from subprocess import call
from transformers import SamModel, SamProcessor

from omegaconf import open_dict
from hydra import compose, initialize
import cv2
import torch
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path, BaseModel

from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from cutie.inference.utils.args_utils import get_dataset_cfg
from gui.interactive_utils import (
    image_to_torch,
    torch_prob_to_numpy_mask,
    index_numpy_to_one_hot_torch,
    overlay_davis,
    color_map_np,
)


class ModelOutput(BaseModel):
    masked_out: Path
    inpaint_out: Optional[Path]
    first_mask_with_SAM: Optional[Path]


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

        # segment-anything
        cache_dir = "weights"
        sam_name = "facebook/sam-vit-huge"
        # add local_file_only if the model is already loaded to cache_dir
        self.sam_processor = SamProcessor.from_pretrained(
            sam_name, cache_dir=cache_dir, local_files_only=True
        )
        self.sam_model = SamModel.from_pretrained(
            sam_name, cache_dir=cache_dir, local_files_only=True
        ).to("cuda")
        self.sam_model.eval()

    def predict(
        self,
        video: Path = Input(description="Input video"),
        mask: Path = Input(
            description="Provide the mask for the first frame. You can leave this blank and use SAM to generate mask below.",
            default=None,
        ),
        mask_with_SAM: str = Input(
            description="Use SAM to generate mask, ignored if a mask_file is provided above. Provide coordinates of the obejct of interest in the format of `x | y`.",
            default=None,
        ),
        max_frames: int = Input(
            description="Number of frames to process. Leave this blank to process the entire video.",
            default=None,
        ),
        inpaint_with_propainter: bool = Input(
            description="Remove the masked objects (inpaint) with ProPainter",
            default=False,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        device = "cuda"
        torch.cuda.empty_cache()

        coord = None
        if mask is None:
            # generate mask for the first frame with SAM
            sam_out = Path(tempfile.mkdtemp()) / "sam_mask.png"
            assert (
                mask_with_SAM is not None
            ), "Please provide a mask file the coordinate of the object to generate the mask with SAM."
            try:
                coord = [float(coord.strip()) for coord in mask_with_SAM.split("|")]
                print(f"Will use SAM to segment object at {coord}.")
            except (IndexError, ValueError):
                raise ValueError(
                    "Could not extract valid x, y values from the mask_with_SAM."
                )
        else:
            sam_out = None
            mask = np.array(Image.open(str(mask)))
            print(f"Masks detected: {np.unique(mask)}")
            num_objects = len(np.unique(mask)) - 1
            assert num_objects > 0, "No object detected, please provide a valid mask."

        # read the input video
        cap = cv2.VideoCapture(str(video))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        current_frame_index = 0
        num_digits = 6
        image_list = []

        out_mask_dir = "out_mask_dir"
        frames_dir = "frames_dir"
        inpaint_out_dir = "inpaint_out_dir"
        for img_dir in [out_mask_dir, frames_dir, inpaint_out_dir]:
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)
            os.makedirs(img_dir)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                while cap.isOpened():
                    # load frame-by-frame
                    _, frame = cap.read()
                    if frame is None or (
                        max_frames is not None and current_frame_index >= max_frames
                    ):
                        break

                    if current_frame_index == 0 and coord is not None:
                        # generate mask for the first frame with SAM
                        inputs = self.sam_processor(
                            frame, input_points=[[coord]], return_tensors="pt"
                        ).to("cuda")
                        outputs = self.sam_model(**inputs)
                        masks = self.sam_processor.image_processor.post_process_masks(
                            outputs.pred_masks,
                            inputs["original_sizes"],
                            inputs["reshaped_input_sizes"],
                        )
                        mask = masks[0].detach().cpu().numpy()[0, 0]
                        Image.fromarray(mask).save(str(sam_out))

                        print(f"Masks detected: {np.unique(mask)}")
                        num_objects = len(np.unique(mask)) - 1
                        assert (
                            num_objects > 0
                        ), "No object detected, please provide valid coordinates."

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

                    mask_out = color_map_np[prediction]

                    padded_idx = str(current_frame_index).zfill(num_digits)

                    Image.fromarray(mask_out).save(
                        f"{out_mask_dir}/mask_{padded_idx}.png"
                    )
                    Image.fromarray(frame).save(f"{frames_dir}/frame_{padded_idx}.png")

                    current_frame_index += 1

        # masked_out = "masked_out.mp4"

        masked_out = Path(tempfile.mkdtemp()) / "masked_out.mp4"
        if inpaint_with_propainter:
            inpaint_out = Path(tempfile.mkdtemp()) / "inpaint_out.mp4"
            # git clone https://github.com/sczhou/ProPainter.git beforehand
            command = (
                "python ProPainter/inference_propainter.py --video "
                + frames_dir
                + " --mask "
                + out_mask_dir
                + " --output "
                + inpaint_out_dir
            )
            call(command, shell=True)

            shutil.copy(
                f"{inpaint_out_dir}/{frames_dir}/inpaint_out.mp4", str(inpaint_out)
            )
            shutil.copy(
                f"{inpaint_out_dir}/{frames_dir}/masked_in.mp4", str(masked_out)
            )

            print("Inpainting finished!")

            return ModelOutput(
                masked_out=masked_out,
                first_mask_with_SAM=sam_out if sam_out is not None else None,
                inpaint_out=inpaint_out,
            )

        writer = imageio.get_writer(masked_out, format="FFMPEG", fps=frame_rate)
        for frame in image_list:
            writer.append_data(frame)
        writer.close()
        return ModelOutput(
            masked_out=masked_out,
            first_mask_with_SAM=sam_out if sam_out is not None else None,
        )
