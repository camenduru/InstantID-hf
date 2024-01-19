import os
import cv2
import math
import random
import numpy as np
from PIL import Image

from diffusers.utils import load_image

import gradio as gr

# global variable
MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def upload_example_to_gallery(images, prompt, style, negative_prompt):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def remove_tips():
    return gr.update(visible=False)

def generate_image(face_image, pose_image, prompt, negative_prompt, num_steps, identitynet_strength_ratio, adapter_strength_ratio, num_outputs, guidance_scale, seed, progress=gr.Progress(track_tqdm=True)):

    if face_image is None:
        raise gr.Error(f"Cannot find any input face image! Please upload the face image")

    face_image = load_image(face_image[0])

    return [face_image], gr.update(visible=True)

### Description
title = r"""
<h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

How to use:<br>
1. Upload a person image or cropped face image. For multiple person images, we will only detect the biggest face. Make sure face is in good condition and not significantly blocked or blurred.
2. (Optionally) upload another person image as reference pose. If not uploaded, we will use the first person image to extract landmarks. 
3. Enter a text prompt as normal text-to-image model.
4. Click the <b>Submit</b> button to start customizing.
5. Share your customizd photo with your friends, enjoy😊!
"""

article = r"""
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{wang2024instantid,
  title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
  author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2401.07519},
  year={2024}
}
```

📧 **Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out at <b>haofanwang.ai@gmail.com</b>.
"""

tips = r"""
### Usage tips of InstantID
1. If you're not satisfied with the similarity, scroll down to "Advanced Options" and increase the weight of "IdentityNet Strength" and "Adapter Strength".
2. If you feel that the saturation is too high, first decrease the Adapter strength. If it is still too high, then decrease the IdentityNet strength.
3. If you find that text control is not as expected, decrease Adapter strength.
"""

css = '''
.gradio-container {width: 85% !important}
'''
with gr.Blocks(css=css) as demo:

    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            
            # upload face image
            face_files = gr.Files(
                        label="Upload a photo of your face",
                        file_types=["image"]
                    )
            uploaded_faces = gr.Gallery(label="Your images", visible=False, columns=1, rows=1, height=512)
            with gr.Column(visible=False) as clear_button_face:
                remove_and_reupload_faces = gr.ClearButton(value="Remove and upload new ones", components=face_files, size="sm")
            
            # optional: upload a reference pose image
            pose_files = gr.Files(
                        label="Upload a reference pose image (optional)",
                        file_types=["image"]
                    )
            uploaded_poses = gr.Gallery(label="Your images", visible=False, columns=1, rows=1, height=512)
            with gr.Column(visible=False) as clear_button_pose:
                remove_and_reupload_poses = gr.ClearButton(value="Remove and upload new ones", components=pose_files, size="sm")
            
            # prompt
            prompt = gr.Textbox(label="Prompt",
                       info="Give simple prompt is enough to achieve good face fedility",
                       placeholder="A photo of a man/woman")
            submit = gr.Button("Submit")

            with gr.Accordion(open=False, label="Advanced Options"):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="low quality",
                    value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                )
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=30,
                )
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.65,
                )
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength",
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.30,
                )
                num_outputs = gr.Slider(
                    label="Number of output images",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=2,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

        with gr.Column():
            gallery = gr.Gallery(label="Generated Images")
            usage_tips = gr.Markdown(label="Usage tips of InstantID", value=tips ,visible=False)

        face_files.upload(fn=swap_to_gallery, inputs=face_files, outputs=[uploaded_faces, clear_button_face, face_files])
        pose_files.upload(fn=swap_to_gallery, inputs=pose_files, outputs=[uploaded_poses, clear_button_pose, pose_files])

        remove_and_reupload_faces.click(fn=remove_back_to_files, outputs=[uploaded_faces, clear_button_face, face_files])
        remove_and_reupload_poses.click(fn=remove_back_to_files, outputs=[uploaded_poses, clear_button_pose, pose_files])

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,            
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[face_files, pose_files, prompt, negative_prompt, num_steps, identitynet_strength_ratio, adapter_strength_ratio, num_outputs, guidance_scale, seed],
            outputs=[gallery, usage_tips]
        )

    gr.Markdown(article)
    
demo.launch()