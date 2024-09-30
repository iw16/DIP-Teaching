import cv2
import gradio as gr
import numpy as np
from numpy.typing import NDArray

# 初始化全局变量，存储控制点和目标点
points_src: list[list[int]] = []
points_dst: list[list[int]] = []
image: NDArray[np.uint8] | None = None

# 上传图像时清空控制点和目标点
def upload_image(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData) -> NDArray[np.uint8]:
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(
    image: NDArray[np.uint8],
    source_pts: NDArray[np.int32],
    target_pts: NDArray[np.int32],
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> NDArray[np.uint8]:
    """ 
    Return
    ------
        A deformed image.
    """
    print(source_pts)
    print(target_pts)
    ### FILL: 基于MLS or RBF 实现 image warping
    if not target_pts.tolist():
        return image
    row, col, chn = image.shape
    warped_image: NDArray[np.uint8] = np.zeros((row, col, chn), dtype=np.uint8)
    ys, xs = np.meshgrid(
        np.arange(row, dtype=np.int16),
        np.arange(col, dtype=np.int16),
        indexing='ij',
    )
    vs: NDArray[np.int16] = np.stack([ys.ravel(), xs.ravel()], axis=-1)
    del xs, ys
    ws: NDArray[np.float32] = np.sum(
        np.square(vs[:, np.newaxis, :] - source_pts[np.newaxis, :, :]) ** -alpha,
        axis=2,
    )
    print(ws)
    pas: NDArray[np.float32] = np.sum(ws[..., np.newaxis] * source_pts[np.newaxis, ...], axis=1)
    qas: NDArray[np.float32] = np.sum(ws[..., np.newaxis] * target_pts[np.newaxis, ...], axis=1)
    phs: NDArray[np.float32] = source_pts[np.newaxis, ...] - pas[:, np.newaxis, ...]
    qhs: NDArray[np.float32] = target_pts[np.newaxis, ...] - qas[:, np.newaxis, ...]
    wpqs: NDArray[np.float32] = np.einsum('ij,ijk,ijl->ikl', ws, phs, qhs)
    wpps: NDArray[np.float32] = np.einsum('ij,ijk,ijl->ikl', ws, phs, phs)
    mat_ms: NDArray[np.float32] = np.einsum('ijk,ikl->ijl', np.linalg.inv(wpps), wpqs)
    fa_vs: NDArray[np.float32] = np.einsum('ij,ijk->ik', vs - pas, mat_ms) + qas
    fa_vs = np.round(fa_vs).astype(np.int16)
    idx_r: NDArray[np.int16] = fa_vs[:, 0].reshape((row, col)).clip(0, row - 1)
    idx_c: NDArray[np.int16] = fa_vs[:, 1].reshape((row, col)).clip(0, col - 1)
    warped_image[idx_r, idx_c] = image
    warped_image[target_pts[:, 1], target_pts[:, 0]] = image[source_pts[:, 1], source_pts[:, 0]]
    return warped_image
    for y in range(row):
        for x in range(col):
            fa_v: NDArray[np.int32] = np.array([x, y], dtype=np.int32)
            if fa_v in target_pts:
                continue
            ws: NDArray[np.float64] = np.sum(np.square(source_pts - fa_v), axis=1) ** alpha
            p_ast: NDArray[np.float64] = np.average(source_pts, axis=0, weights=ws)
            q_ast: NDArray[np.float64] = np.average(target_pts, axis=0, weights=ws)
            p_hat: NDArray[np.float64] = source_pts - p_ast
            q_hat: NDArray[np.float64] = target_pts - q_ast
            s_wpq: NDArray[np.float64] = np.sum([
                w * (p_i.reshape((2, 1)) @ q_i.reshape((1, 2))) for w, p_i, q_i in zip(ws, p_hat, q_hat)
            ], axis=0)
            s_wpp: NDArray[np.float64] = np.sum([
                w * (p_i.reshape((2, 1)) @ p_i.reshape((1, 2))) for w, p_i in zip(ws, p_hat)
            ], axis=0)
            #s_wpq: NDArray[np.float64] = np.sum(ws * np.matmul(p_hat.reshape((-1, 2, 1)), q_hat.reshape((-1, 1, 2)), axes=[1, 2]), axis=0)
            #s_wpp: NDArray[np.float64] = np.sum(ws * np.matmul(p_hat.reshape((-1, 2, 1)), p_hat.reshape((-1, 1, 2)), axes=[1, 2]), axis=0)
            del p_hat, q_hat
            mat_m: NDArray[np.float64] = np.linalg.inv(s_wpq) @ s_wpp
            del s_wpq, s_wpp
            v: NDArray[np.float64] = (fa_v - q_ast) @ mat_m + q_ast
            del mat_m, p_ast, q_ast
            v = np.round(v.clip(min=0, max=(row - 1, col - 1))).astype(np.int32)
            warped_image[y, x] = image[*v]
    return warped_image

def run_warping() -> NDArray[np.uint8]:
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points() -> NDArray[np.uint8]:
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)

if __name__ == '__main__':
    # 启动 Gradio 应用
    demo.launch()
