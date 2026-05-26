StreamVLN 具身导航复现与模型推理指南

综述由AI生成介绍 StreamVLN 流式视觉语言导航模型的复现流程。内容包括创建 Conda 环境、安装 Habitat 仿真环境、准备 Matterport3D 及 VLN-CE 数据集、下载预训练模型权重。详细说明了多 GPU 和单 GPU 下的评估推理命令，展示了可视化效果代码修改，并提供了分布式训练指令。适用于具身智能导航任务的研究与部署。
锁机制
发布于 2026/4/6
更新于 2026/5/12
29 浏览
StreamVLN 具身导航复现与模型推理指南

StreamVLN 通过在线、多轮对话的方式，输入连续视频，输出动作序列。

通过结合语言指令、视觉观测和空间位姿信息，驱动模型生成导航动作（前进、左转、右转、停止）。

论文地址：StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling

代码地址：https://github.com/OpenRobotLab/StreamVLN

下面是示例效果：

文章配图
1、创建 Conda 环境

首先创建一个 Conda 环境，名字为 streamvln，python 版本为 3.9；

然后进入 streamvln 环境，执行下面命令：

conda create -n streamvln python=3.9
conda activate streamvln

2、安装 habitat 仿真环境

先安装 habitat-sim，执行下面命令进行安装

conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat

再安装 habitat-lab，

git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab # install habitat_lab
pip install -e habitat-baselines # install habitat_baselines

3、安装第三方的依赖库

获取 StreamVLN 的代码

git clone https://github.com/OpenRobotLab/StreamVLN.git
cd StreamVLN

安装其他依赖库：

pip install -r requirements.txt

2025/7/23 补丁安装：需要安装 protobuf==3.20.1

pip install protobuf==3.20.1

4、准备数据集

需要准备三种类型的数据，新建一个 data 文件夹来存放：

1）Matterport3D (MP3D) Scenes

快速下载地址：https://cloud.tsinghua.edu.cn/f/03e0ca1430a344efa72b/?dl=1

文章配图

每个文件夹中，包含一个.glb 文件：

文章配图

如果是想要完整 MP3D 数据，推荐使用'批量下载'方式～（可选）

2）VLN-CE Episodes

下载 VLN-CE episodes 的链接，然后重命名：

    r2r（重命名 R2R_VLNCE_v1/ -> r2r/）
    rxr（重命名 RxR_VLNCE_v0/ -> rxr/）
    envdrop（重命名 R2R_VLNCE_v1-3_preprocessed/envdrop/ -> envdrop/）

最后，将它们解压到 data/datasets/ 目录中。

3）Collected Trajectory Data

作者提供预先收集的观察 - 动作轨迹数据用于训练；

这些轨迹是在 Matterport3D 环境下使用 R2R 和 RxR 的训练片段收集的。

下载链接：https://huggingface.co/datasets/cywan/StreamVLN-Trajectory-Data/blob/main/README.md

文章配图

下载好上面三个数据集后，文件夹结构应如下所示：

    data/ ├── datasets/ │ ├── r2r/ │ │ ├── train/ │ │ ├── val_seen/ │ │ │ └── val_seen.json.gz │ │ └── val_unseen/ │ │ └── val_unseen.json.gz │ ├── rxr/ │ │ ├── train/ │ │ ├── val_seen/ │ │ │ ├── val_seen_guide.json.gz │ │ │ └── ... │ │ └── val_unseen/ │ │ ├── val_unseen_guide.json.gz │ │ └── ... │ └── envdrop/ │ ├── envdrop.json.gz │ └── ... │ ├── scene_datasets/ │ └── mp3d/ │ ├── 17DRP5sb8fy/ │ ├── 1LXtFkjw3qL/ │ └── ... └── trajectory_data/ ├── R2R/ │ ├── images/ │ └── annotations.json ├── RxR/ │ ├── images/ │ └── annotations.json └── EnvDrop/ ├── images/ └── annotations.json

5、下载模型权重

提供了两个模型权重：

模型权重 1：基准测试重现（仿真环境）

使用此权重来重现 VLN-CE 基准测试的结果，链接：

https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln

文章配图

下载好的模型权重，存放在 data 目录下，比如：

data/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln

模型权重 2：真实世界部署

下载链接：https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world

做了两处修改：

    删除多余的初始转弯动作：为了更好地对齐指令，删除了指令中未提及的初始左/右转弯。
    轨迹安全：增强的避障功能可确保在现实环境中更可靠的导航。

文章配图

为测试实际场景适用性，在真实环境中用 Unitree Go2 机器狗 部署 StreamVLN：

    件配置：机器人搭载 Intel RealSense D455 RGB-D 相机采集视觉数据，推理任务部署在远程工作站（配备 RTX 4090 GPU），实现'机器人采集数据→服务器推理→机器人执行动作'的闭环。
    延迟表现：单次推理平均延迟 0.27 秒（生成 4 个动作），室内通信延迟 0.2 秒，室外 1.0 秒，总延迟满足实时导航需求（类似'机器人看到环境后，能快速计算下一步动作，不会卡顿'）。

6、模型评估推理

修改 StreamVLN-main/scripts/streamvln_eval_multi_gpu.sh

模型权重路径：CHECKPOINT="data/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"

1）多 GPU 评估推理

修改 streamvln_eval_multi_gpu.sh 为：

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet MASTER_PORT=$((RANDOM % 101 + 20000)) CHECKPOINT="data/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln" echo "CHECKPOINT: ${CHECKPOINT}" torchrun --nproc_per_node=4 --master_port=$MASTER_PORT streamvln/streamvln_eval.py --model_path $CHECKPOINT 

其中的--nproc_per_node=4，根据具体的显卡数量来修改

执行命令：

sh scripts/streamvln_eval_multi_gpu.sh

打印信息：

    CHECKPOINT: data/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln [2025-07-23 19:02:56,669] torch.distributed.run: [WARNING] [2025-07-23 19:02:56,669] torch.distributed.run: [WARNING] ************** [2025-07-23 19:02:56,669] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. [2025-07-23 19:02:56,669] torch.distributed.run: [WARNING] ************** | distributed init (rank 0): env://, gpu 0 | distributed init (rank 1): env://, gpu 1 | distributed init (rank 2): env://, gpu 2 | distributed init (rank 3): env://, gpu 3 Sliding Window Attention is enabled but not implemented for eager; unexpected results may be encountered. ... (省略部分日志)

显存占用情况：

    | 0 N/A N/A 1507162 C+G .../envs/streamvln/bin/python3.9 29085MiB | | 0 N/A N/A 1507163 G .../envs/streamvln/bin/python3.9 825MiB | | 0 N/A N/A 1507164 G .../envs/streamvln/bin/python3.9 825MiB | | 0 N/A N/A 1507165 G .../envs/streamvln/bin/python3.9 825MiB | | 1 N/A N/A 1507163 C .../envs/streamvln/bin/python3.9 36670MiB | | 2 N/A N/A 1507164 C .../envs/streamvln/bin/python3.9 36982MiB | | 3 N/A N/A 1507165 C .../envs/streamvln/bin/python3.9 32280MiB |

输出结果：

    [19:13:20.799677] 32 You are an autonomous navigation assistant. Your task is to Go straight through the doorway. Go to the left and then left again till you see the star burst pattern on the floor Go through the bedroom doorway and stop when you get to the bed. Wait there. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP. These are your historical observations .

    [19:13:21.780076] <|im_start|>assistant ↑↑↑→<|im_end|> [19:13:21.780184] 解析的动作序列 [1, 1, 1, 3] ...

官方效果 - 原版代码可视化：

文章配图

改进版：（streamvln/streamvln_eval.py）

# 导入系统相关库
import sys
import os
# 将上级目录添加到系统路径，便于导入自定义模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 导入正则表达式、进度条、PyTorch 等工具库
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import transformers
import numpy as np
# 导入类型注解、配置工具、图像处理等库
from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageFile, ImageDraw, ImageFont
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
# 导入深度图像过滤函数
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array
# 导入 Habitat 环境相关库（用于导航模拟）
import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
# 导入自定义模型和工具函数
from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import dict_to_cuda
from utils.dist import *
# 分布式处理工具
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX

class VLNEvaluator:
    """视觉语言导航 (VLN) 评估器类，用于评估模型在 Habitat 环境中的导航性能"""
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        tokenizer: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda')
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )
        print(f"config 类型 = {type(self.config)}")
        print(OmegaConf.to_yaml(self.config))
        self._camera_height = self.sim_sensors_config.rgb_sensor.position[1]
        self._min_depth = self.sim_sensors_config.depth_sensor.min_depth
        self._max_depth = self.sim_sensors_config.depth_sensor.max_depth
        camera_fov_rad = np.deg2rad(self.sim_sensors_config.depth_sensor.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        self.image_processor = model.get_vision_tower().image_processor
        self.model = model
        self.tokenizer = tokenizer
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history

    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        target_height = self.image_processor.crop_size['height']
        target_width = self.image_processor.crop_size['width']
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
        return img, (target_width, target_height)

    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        intrinsic_matrix = np.array([
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        return intrinsic_matrix

    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]
        intrinsic[:, 0] /= ori_size[0] / target_size[0]
        intrinsic[:, 1] /= ori_size[1] / target_size[1]
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2
        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)
        return intrinsic

    def get_axis_align_matrix(self):
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
        return ma

    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def eval_action(self, idx) -> None:
        env = self.config_env()
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)
        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
        if get_rank() == 0:
            sucs.append(res['success'])
            spls.append(res['spl'])
            oss.append(res['os'])
            ones.append(res['ne'])
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"当前场景 ID = {scene_id}")
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"场景 {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                print("开始 episode：", episode_instruction)
                episode_id = episode.episode_id
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                self.model.reset_for_env(idx)
                env.current_episode = episode
                observations = env.reset()
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                vis_frames = []
                step_id = 0
                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]
                rgb_list = []
                depth_list = []
                depth_images_list = []
                pose_list = []
                intrinsic_list = []
                time_ids = []
                action_seq = []
                past_key_values = None
                output_ids = None
                while not env.episode_over:
                    self.model.eval()
                    time_ids.append(step_id)
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000
                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height
                    camera_position = np.array([x, -y, self._camera_height + height])
                    robot_xy = camera_position[:2]
                    tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                    rotation = agent_state.rotation
                    translation = agent_state.position
                    rotation_matrix = quaternion.as_rotation_matrix(rotation)
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = translation
                    image = Image.fromarray(rgb).convert('RGB')
                    image_size = image.size
                    image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                    depth_image, resize_shape = self.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)
                    intrinsic = self.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                    intrinsic = torch.from_numpy(intrinsic).float()
                    rgb_list.append(image)
                    depth_list.append(torch.from_numpy(depth_image).float())
                    pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())
                    intrinsic_list.append(intrinsic)
                    episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                    info = env.get_metrics()
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        frame_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(frame_pil)
                        img_width, img_height = frame_pil.size
                        task_text = f"## Task: {episode_instruction}"
                        metrics = env.get_metrics()
                        result_text = (
                            f"scene_episode ID: {scene_id}_{episode_id}\n"
                            f"oracle_success = {metrics['oracle_success']:.1f}\n"
                            f"distance_to_goal = {metrics['distance_to_goal']:.2f}"
                        )
                        full_text = f"{task_text}\n{result_text}"
                        base_font_size = int(img_height * 0.04)
                        margin = int(img_height * 0.015)
                        line_spacing = int(base_font_size * 0.3)
                        text_color = (255, 255, 255)
                        bg_color = (0, 0, 0, 200)
                        try:
                            font = ImageFont.truetype("simhei.ttf", base_font_size)
                        except:
                            font = ImageFont.load_default()
                        max_line_width = int(img_width * 0.4)
                        def wrap_text(text, font, max_width):
                            lines = []
                            current_line = ""
                            for char in text:
                                if char == "\n":
                                    lines.append(current_line)
                                    current_line = ""
                                    continue
                                test_line = current_line + char
                                bbox = draw.textbbox((0, 0), test_line, font=font)
                                if (bbox[2] - bbox[0]) > max_width:
                                    lines.append(current_line)
                                    current_line = char
                                else:
                                    current_line = test_line
                            if current_line:
                                lines.append(current_line)
                            return lines
                        wrapped_lines = wrap_text(full_text, font, max_line_width)
                        line_height = base_font_size + line_spacing
                        total_text_height = (len(wrapped_lines) * line_height) - line_height
                        max_line_width_actual = 0
                        for line in wrapped_lines:
                            bbox = draw.textbbox((0, 0), line, font=font)
                            line_width = bbox[2] - bbox[0]
                            if line_width > max_line_width_actual:
                                max_line_width_actual = line_width
                        bg_x1 = margin
                        bg_y1 = margin
                        bg_x2 = bg_x1 + max_line_width_actual + 2 * margin
                        bg_y2 = bg_y1 + total_text_height + 2 * margin
                        if bg_y2 > img_height - margin:
                            bg_y2 = img_height - margin
                        if bg_x2 > img_width * 0.4 + margin:
                            bg_x2 = int(img_width * 0.4) + margin
                        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
                        text_x = bg_x1 + margin
                        text_y = bg_y1 + margin
                        for line in wrapped_lines:
                            if text_y + base_font_size > bg_y2 - margin:
                                break
                            draw.text((text_x, text_y), line, font=font, fill=text_color)
                            text_y += line_height
                        frame = np.array(frame_pil)
                        vis_frames.append(frame)
                    if len(action_seq) == 0:
                        if output_ids is None:
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                            if step_id != 0 :
                                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', episode.instruction.instruction_text)
                            add_system = True
                            print(step_id, sources[0]["value"])
                        else:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                        input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                        if output_ids is not None:
                            input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)
                        images = rgb_list[-1:]
                        depths = depth_list[-1:]
                        poses = pose_list[-1:]
                        intrinsics = intrinsic_list[-1:]
                        if step_id != 0 and step_id % self.num_frames == 0:
                            if self.num_history is None:
                                history_ids = slice(0, time_ids[0], self.num_future_steps)
                            else:
                                history_ids = slice(0, time_ids[0], (time_ids[0] // self.num_history))
                            images = rgb_list[history_ids] + images
                            depths = depth_list[history_ids] + depths
                            poses = pose_list[history_ids] + poses
                            intrinsics = intrinsic_list[history_ids] + intrinsics
                        input_dict = {
                            'images': torch.stack(images).unsqueeze(0),
                            'depths': torch.stack(depths).unsqueeze(0),
                            'poses': torch.stack(poses).unsqueeze(0),
                            'intrinsics': torch.stack(intrinsics).unsqueeze(0),
                            'inputs': input_ids,
                            'env_id': idx,
                            'time_ids': [time_ids],
                            'task_type': [0]
                        }
                        input_dict = dict_to_cuda(input_dict, self.device)
                        for key, value in input_dict.items():
                            if key in ['images', 'depths', 'poses', 'intrinsics']:
                                input_dict[key] = input_dict[key].to(torch.bfloat16)
                        outputs = self.model.generate(
                            **input_dict,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=10000,
                            use_cache=True,
                            return_dict_in_generate=True,
                            past_key_values=past_key_values
                        )
                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values
                        llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        print(llm_outputs, flush=True)
                        action_seq = self.parse_actions(llm_outputs)
                        print('解析的动作序列', action_seq, flush=True)
                        if len(action_seq) == 0:
                            action_seq = [0]
                        action = action_seq.pop(0)
                        observations = env.step(action)
                        step_id += 1
                    if step_id % self.num_frames == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []
                    process_bar.update(1)
                    metrics = env.get_metrics()
                    if self.save_video:
                        images_to_video(
                            vis_frames,
                            os.path.join(self.output_path, f'vis_{self.epoch}'),
                            f'{scene_id}_{episode_id}',
                            fps=6,
                            quality=9
                        )
                        vis_frames.clear()
                    sucs.append(metrics['success'])
                    spls.append(metrics['spl'])
                    oss.append(metrics['oracle_success'])
                    ones.append(metrics['distance_to_goal'])
                    print(f"场景-episode {scene_id}_{episode_id} 结果：成功={metrics['success']}, SPL={metrics['spl']}, Oracle 成功={metrics['oracle_success']}, 到目标距离={metrics['distance_to_goal']}")
                    result = {
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "success": metrics["success"],
                        "spl": metrics["spl"],
                        "os": metrics['oracle_success'],
                        "ne": metrics["distance_to_goal"],
                        "steps": step_id,
                        "episode_instruction": episode_instruction
                    }
                    with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                        f.write(json.dumps(result) + "\n")
                env.close()
                return (torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device))

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.", add_system: bool = False):
        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(tokenizer)
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        unmask_tokens_idx = [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids
        chat_template = "{% for message in messages %}{{'<' + message['role'] + '\n' + message['content'] + '>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
            input_id = []
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
            for conv in source:
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]
                role = roles.get(role, role)
                conv = [{"role" : role, "content" : content}]
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations

    def pad_tensors(tensors, lens=None, max_len=None, pad=0):
        if lens is None:
            lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
        if max_len is None:
            max_len = max(lens)
        bs = len(tensors)
        hid = tensors[0].shape[1:]
        dtype = tensors[0].dtype
        output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
        if pad:
            output.data.fill_(pad)
        for i, (t, l) in enumerate(zip(tensors, lens)):
            output.data[i, :l, ...] = t.data
        return output

def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="本地进程排名")
    parser.add_argument("--model_path", type=str,, help="模型路径")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml', help="Habitat 配置文件路径")
    parser.add_argument("--eval_split", type=str, default='val_unseen', help="评估数据集分割")
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln', help="结果输出路径")
    parser.add_argument("--num_future_steps", type=int, default=4, help="未来步骤数")
    parser.add_argument("--num_frames", type=int, default=32, help="每批处理的帧数")
    parser.add_argument("--save_video", default=True, help="是否保存导航视频")
    parser.add_argument("--num_history", type=int, default=8, help="历史帧数")
    parser.add_argument("--model_max_length", type=int, default=4096, help="模型最大序列长度")
    parser.add_argument('--world_size', default=1, type=int, help='分布式进程数')
    parser.add_argument('--rank', default=0, type=int, help='进程排名')
    parser.add_argument('--gpu', default=0, type=int, help='GPU 设备 ID')
    parser.add_argument('--port', default='1111', help='分布式通信端口')
    parser.add_argument('--dist_url', default='env://', help='分布式通信 URL')
    parser.add_argument('--device', default='cuda', help='设备类型')
    args = parser.parse_args()
    init_distributed_mode(args)
    local_rank = args.local_rank
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right"
    )
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(local_rank)
    evaluate(model, tokenizer, args)

def evaluate(model, tokenizer, args):
    model.eval()
    world_size = get_world_size()
    model.reset(world_size)
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank())
    ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
    dist.all_gather(ep_num_all, ep_num)
    sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
    spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
    oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
    ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
    dist.barrier()
    dist.all_gather(sucs_all, sucs)
    dist.all_gather(spls_all, spls)
    dist.all_gather(oss_all, oss)
    dist.all_gather(ones_all, ones)
    dist.barrier()
    sucs_all = torch.cat(sucs_all, dim=0)
    spls_all = torch.cat(spls_all, dim=0)
    oss_all = torch.cat(oss_all, dim=0)
    ones_all = torch.cat(ones_all, dim=0)
    result_all = {
        "平均成功率": (sum(sucs_all)/len(sucs_all)).item(),
        "平均 SPL": (sum(spls_all)/len(spls_all)).item(),
        "平均 Oracle 成功率": (sum(oss_all)/len(oss_all)).item(),
        "平均到目标距离": (sum(ones_all)/len(ones_all)).item(),
        "总 episode 数": len(sucs_all)
    }
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()

可视化效果（实时显示任务、是否成功、到目标的距离）

文章配图

2）单 GPU 评估推理

执行命令：（num_frames 设置小一些，默认 32 需要较大显存）

python streamvln/streamvln_eval.py --model_path "data/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln" --num_frames 8

打印信息：

    (streamvln) lgp@lgp-MS-7E07:/2025_project/StreamVLN-main$ (streamvln) lgp@lgp-MS-7E07:/2025_project/StreamVLN-main$ python streamvln/streamvln_eval.py --model_path "data/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln" Not using distributed mode Sliding Window Attention is enabled but not implemented for eager; unexpected results may be encountered. [19:45:30.773156] The checkpoint seems to contain vision_tower weights: mm_tunable_parts contains mm_vision_tower. config.json: 576B [00:00, 60.4kB/s] model.safetensors: 32%|██████████████████ | 1.14G/3.51G [05:15<16:04, 2.47MB/s] model.safetensors: 100%|████████████████████████████████████████████████████████| 3.51G/3.51G [08:42<00:00, 6.73MB/s] Loading checkpoint shards: 100%|███████████████████████████████████████████████████████| 4/4 [00:00<00:00, 9.20it/s] ............

7、模型训练

使用分布式设置，执行多节点多 GPU 训练，运行指令：

sbatch scripts/streamvln_train_slurm.sh
