import math as m
import typing as tp

import cv2
import gradio as gr
import numpy as np
import PIL.Image
from numpy.typing import NDArray
Matrix: tp.TypeAlias = NDArray

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
_T = tp.TypeVar(name='_T')
def to_3x3(affine_matrix: Matrix[_T]) -> Matrix[_T]:
    dtype: np.dtype = affine_matrix.dtype
    return np.vstack([affine_matrix, [0, 0, 1]], dtype=dtype)

def fit_to_size(
    img: NDArray[np.uint8],
    bounds: tuple[int, int],
    bgcolor: int = 0xffffff,
) -> NDArray[np.uint8]:
    row, col, chn = img.shape
    row_fit, col_fit = bounds
    row_aff: int = min(row, row_fit)
    pi_n: int = (row - row_aff) // 2 # pad_initial_north
    pf_n: int = (row_fit - row_aff) // 2 # pad_final_north
    col_aff: int = min(col, col_fit)
    pi_w: int = (col - col_aff) // 2 # pad_initial_west
    pf_w: int = (col_fit - col_aff) // 2 # pad_final_west
    img_fit: NDArray[np.uint8] = np.zeros((row_fit, col_fit, chn), dtype=np.uint8)
    bg_px: NDArray[np.uint8] = np.array([
        (bgcolor >> 16) & 0xff,
        (bgcolor >> 8) & 0xff,
        bgcolor & 0xff,
    ], dtype=np.uint8)
    img_fit[:, :, :] = bg_px
    img_fit[pf_n:pf_n+row_aff, pf_w:pf_w+col_aff, :] = img[pi_n:pi_n+row_aff, pi_w:pi_w+col_aff, :]
    return img_fit

def apply_scale(
    img: NDArray[np.uint8],
    scale: float,
    bounds: tuple[int, int] | None = None,
    bgcolor: int = 0xffffff,
) -> NDArray[np.uint8]:
    row, col, chn = img.shape
    new_shape = int(row * scale), int(col * scale), chn
    fig: NDArray[np.uint8] = np.zeros(new_shape, dtype=np.uint8)
    # weights & indices
    wt_s, idx_r = np.modf(np.linspace(0, row, new_shape[0], endpoint=False))
    wt_e, idx_c = np.modf(np.linspace(0, col, new_shape[1], endpoint=False))
    wt_s = wt_s.reshape((-1, 1, 1))[:-1]
    wt_e = wt_e.reshape((-1, 1))[:-1]
    wt_n: NDArray[np.float64] = 1 - wt_s
    wt_w: NDArray[np.float64] = 1 - wt_e
    wt_nw: NDArray[np.float64] = wt_n * wt_w
    wt_ne: NDArray[np.float64] = wt_n * wt_e
    wt_se: NDArray[np.float64] = wt_s * wt_e
    wt_sw: NDArray[np.float64] = wt_s * wt_w
    # basic array
    idx_r = idx_r.astype(np.int32).reshape((-1, 1))
    idx_c = idx_c.astype(np.int32)
    basic: NDArray[np.uint8] = img[idx_r, idx_c]
    # partitions
    ori_nw: NDArray[np.uint8] = basic[:-1, :-1]
    ori_ne: NDArray[np.uint8] = basic[:-1, 1:]
    ori_se: NDArray[np.uint8] = basic[1:, 1:]
    ori_sw: NDArray[np.uint8] = basic[1:, :-1]
    # bilinear interpolation
    fig[:-1, :-1] = (ori_nw * wt_nw + ori_ne * wt_ne + ori_se * wt_se + ori_sw * wt_sw).clip(0, 0xff).astype(np.uint8)
    fig[-1, :-1] = (basic[-1, :-1] * wt_w + basic[-1, 1:] * wt_e).clip(0, 0xff).astype(np.uint8)
    fig[:-1, -1] = (basic[:-1, -1] * wt_n.reshape((-1, 1)) + basic[1:, -1] * wt_s.reshape((-1, 1))).clip(0, 0xff).astype(np.uint8)
    if bounds:
        return fit_to_size(fig, bounds, bgcolor)
    else:
        return fig

def apply_rotation(
    img: NDArray[np.uint8],
    degrees: float,
    bounds: tuple[int, int],
    bgcolor: int = 0xffffff,
) -> NDArray[np.uint8]:
    row, col, chn = img.shape
    max_row, max_col = bounds
    fig: NDArray[np.uint8] = np.zeros(shape=(max_row, max_col, 3), dtype=np.uint8)
    xs: NDArray[np.int32] = np.arange(row, dtype=np.float64).reshape((-1, 1))
    ys: NDArray[np.int32] = np.arange(col, dtype=np.float64)
    xs -= row / 2
    ys -= col / 2
    c, s = m.cos(m.radians(degrees)), m.sin(m.radians(degrees))
    wt_s, idx_r = np.modf(row / 2 + c * xs + s * ys)
    wt_e, idx_c = np.modf(col / 2 - s * xs + c * ys)
    wt_s = wt_s[..., np.newaxis]
    wt_e = wt_e[..., np.newaxis]
    wt_n: NDArray[np.float64] = 1 - wt_s
    wt_w: NDArray[np.float64] = 1 - wt_e
    wt_nw: NDArray[np.float64] = wt_n * wt_w
    wt_ne: NDArray[np.float64] = wt_n * wt_e
    wt_se: NDArray[np.float64] = wt_s * wt_e
    wt_sw: NDArray[np.float64] = wt_s * wt_w
    # basic array
    idx_r = np.round(idx_r).astype(np.int32)
    idx_c = np.round(idx_c).astype(np.int32)
    basic: NDArray[np.uint8] = img[idx_r.clip(0, max_row - 1), idx_c.clip(0, max_col - 1)]
    bg_arr: NDArray[np.uint8] = np.array([
        (bgcolor >> 16) & 0xff,
        (bgcolor >> 8) & 0xff,
        bgcolor & 0xff,
    ])
    # for those out-of-bound indices, fill the pixels with bgcolor
    basic[(idx_r < 0) & (idx_r >= max_row) | (idx_c < 0) & (idx_c >= max_col)] = bg_arr
    # partitions
    ori_nw: NDArray[np.uint8] = basic[:-1, :-1]
    ori_ne: NDArray[np.uint8] = basic[:-1, 1:]
    ori_se: NDArray[np.uint8] = basic[1:, 1:]
    ori_sw: NDArray[np.uint8] = basic[1:, :-1]
    # bilinear interpolation
    fig[:-1, :-1] = (ori_nw * wt_nw[:-1, :-1] + ori_ne * wt_ne[:-1, :-1] + ori_se * wt_se[:-1, :-1] + ori_sw * wt_sw[:-1, :-1]).clip(0, 0xff).astype(np.uint8)
    fig[-1, :-1] = (basic[-1, :-1] * wt_w[-1, :-1] + basic[-1, 1:] * wt_e[-1, :-1]).clip(0, 0xff).astype(np.uint8)
    fig[:-1, -1] = (basic[:-1, -1] * wt_n[:-1, -1] + basic[1:, -1] * wt_s[:-1, -1]).clip(0, 0xff).astype(np.uint8)
    return fit_to_size(fig, bounds, bgcolor)

def apply_translation(
    img: NDArray[np.uint8],
    t_x: int,
    t_y: int,
    bounds: tuple[int, int],
    bgcolor: int = 0xffffff,
) -> NDArray[np.uint8]:
    imt: NDArray[np.uint8] = fit_to_size(img, bounds, bgcolor)
    bg_arr: NDArray[np.uint8] = np.array([
        (bgcolor >> 16) & 0xff,
        (bgcolor >> 8) & 0xff,
        bgcolor & 0xff,
    ], dtype=np.uint8)
    # translation_y, south positive
    if t_y > 0:
        imt[t_y:] = imt[:-t_y]
        imt[:t_y] = bg_arr
    elif t_y < 0:
        imt[:-t_y] = imt[t_y:]
        imt[-t_y:] = bg_arr
    # translation_x, east positive
    if t_x > 0:
        imt[:, t_x:] = imt[:, :-t_x]
        imt[:, :t_x] = bg_arr
    elif t_x < 0:
        imt[:, :-t_x] = imt[:, t_x:]
        imt[:, -t_x:] = bg_arr
    return imt

def apply_flip_horizontal(
    img: NDArray[np.uint8],
    flip: bool = False,
) -> NDArray[np.uint8]:
    if flip:
        return img[:, ::-1]
    else:
        return img

# Function to apply transformations based on user inputs
def apply_transform(
    img: NDArray[np.uint8],
    scale: float,
    rotation: float,
    translation_x: int,
    translation_y: int,
    flip_horizontal: bool,
) -> NDArray[np.uint8]:
    row, col, chn = img.shape
    # Pad the image to avoid boundary issues
    pad: int = min(row, col) // 2
    row_new: int = pad * 2 + row
    col_new: int = pad * 2 + col
    img_new: NDArray[np.uint8] = 0xff * np.ones((row_new, col_new, chn), dtype=np.uint8)
    img_new[pad:pad+row, pad:pad+col] = img

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    # rotation and scaling transformations are commutative
    # scaling
    img_new = apply_scale(img, scale, bounds=(row_new, col_new))
    # rotation
    img_new = apply_rotation(img_new, degrees=rotation, bounds=(row_new, col_new))
    # translations
    img_new = apply_translation(img_new, translation_x, translation_y, bounds=(row_new, col_new))
    # flip
    img_new = apply_flip_horizontal(img_new, flip_horizontal)
    return img_new

# Gradio Interface
def interactive_transform() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
if __name__ == '__main__':
    interactive_transform().launch()
