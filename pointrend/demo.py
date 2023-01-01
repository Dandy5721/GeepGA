#coding=utf-8

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''
Bo
'''

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

import os
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    verify_results,
)

from point_rend import add_pointrend_config

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo



# constants
WINDOW_NAME = "COCO detections"
CLASS_NAMES = ["__background__","BN-Skull-Out",'BN-Skull-In',
               'BN-CSP','BN-Cerebellar','BN-CM','Add-Ruler',
               'BN-NF','BN-Lateral-Ventricle',
               'PP-Abdomen-Axial-Planes','LN-Femur']

DATASET_ROOT = 'C:\\Users\\aiyunji-xj\\detectron2\\projects\\PointRend\\datasets\\myself'
ANN_ROOT = os.path.join(DATASET_ROOT,'')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'val')

TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
# VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_my_val": (VAL_PATH, VAL_JSON),
}
def plain_register_dataset():
    # 训练集
    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                             evaluator_type='coco',  # 指定评估方式
                                             json_file=TRAIN_JSON,
                                             image_root=TRAIN_PATH)

    # 验证/测试集
    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                           evaluator_type='coco',  # 指定评估方式
                                           json_file=VAL_JSON,
                                           image_root=VAL_PATH)


'''
detectron2/data/datasets/builtin.py 包含所有数据集结构
detectron2/data/datasets/builtin_meta.py 包含每个数据集元数据
detectron2/data/transforms/transform_gen.py 包含基本的数据增强
detectron2/config/defaults.py 包含所有模型参数
'''


# args传参 -- cfg获取 -- VisualizationDemo模型建立 (元数据获取 -- \
#   DefaultPredictor预测模型 (元结构注册 -- 指定要评估 -- 元数据注册 -- 加载模型权重 -- 图像resize和BGR)) -- \
#   预测数据读入 -- VisualizationDemo预测结果及可视化 (DefaultPredictor调用模型和BGR图像预测 -- visualizer画图)

# MetadataCatalog 含有数据集的元数据, 比如COCO的类别；全局变量,禁止滥用.
# DatasetCatalog 保留了用于获取数据集的方法.

# 基于demo.py 写模型运行main函数
# 基于config/defaults.py 改所新模型包含的参数
# 基于.yaml 改defaults所包含参数的参数值
# 包含所有用到的函数的子函数 builtins.py

# 向默认.yaml里读入参数
def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    args.config_file = "C:\\Users\\aiyunji-xj\\detectron2\\projects\\PointRend\\configs\\InstanceSegmentation\\pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    cfg.merge_from_file(args.config_file)  # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)  # 从CLI参数 覆盖配置

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("coco_my_train",)  # 训练数据集名称
    cfg.DATASETS.TEST = ("coco_my_val",)
    cfg.DATALOADER.NUM_WORKERS = 4  # 单线程

    #    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 640  # 训练图片输入的最大尺寸
    cfg.INPUT.MAX_SIZE_TEST = 640  # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 768)  # 训练图片输入的最小尺寸，可以设定为多尺度训练
    cfg.INPUT.MIN_SIZE_TEST = 640
    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING，其存在两种配置，分别为 choice 与 range ：
    # range 让图像的短边从 512-768随机选择
    # choice ： 把输入图像转化为指定的，有限的几种图片大小进行训练，即短边只能为 512或者768
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'

    cfg.MODEL.RETINANET.NUM_CLASSES = 11  # 类别数+1（因为有background）
    # cfg.MODEL.WEIGHTS="/home/yourstorePath/.pth"
    cfg.SOLVER.IMS_PER_BATCH = 4  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size

    # 根据训练数据总数目以及batch_size，计算出每个epoch需要的迭代次数
    # 9000为你的训练数据的总数目，可自定义
    ITERS_IN_ONE_EPOCH = int(1000 / cfg.SOLVER.IMS_PER_BATCH)

    # 指定最大迭代次数
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1  # 12 epochs，
    # 初始学习率
    cfg.SOLVER.BASE_LR = 0.002
    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9
    # 权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1
    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"
    # 保存模型文件的命名数据减1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # 迭代到指定次数，进行一次评估
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    # cfg.TEST.EVAL_PERIOD = 100


    # 设置模型负样本判定阈值
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg



# 调用demo脚本所需要设定的参数, 会传入setup_cfg(args)
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    # 传入.yaml文件,对应各种参数配置
    # configs/Base-RCNN-FPN.yaml --> configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
    parser.add_argument(
        "--config-file",
        default="C:\\Users\\aiyunji-xj\\detectron2\\projects\\PointRend\\configs\\InstanceSegmentation\\pointrend_rcnn_R_50_FPN_3x_coco.yaml",
        metavar="FILE",
        help="path to config file",
    )
    # 是否读取摄像头
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    # 是否读取视频
    parser.add_argument("--video-input", help="Path to video file.")
    # 预测样本 --input input1.jpg input2.jpg
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    # 结果输出路径 output/
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    # 置信度阈值
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    # 其余任何没指定的参数
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    # 多进程启动
    # mp.set_start_method("spawn", force=True)
    # 获取参数
    input="C:\\Users\\aiyunji-xj\\Desktop\\newGT"
    filelist=os.listdir(input)


    args = get_parser().parse_args()
    # print("args",args)
    args.input=[]
    for file in filelist:
        if file.endswith(".jpg"):
            args.input.append(os.path.join(input,file))

    args.output="C:\\Users\\aiyunji-xj\\Desktop\\GTresult"
    args.confidence_threshold=0.85
    args.opts.append("MODEL.WEIGHTS")
    args.opts.append("weights/model_final.pth")
    print("args", args)
    # 日志
    setup_logger(output="logs/", name="fvcore")
    logger = setup_logger()
    # logger.info("Arguments: " + str(args))

    # 向默认.yaml里读入参数, 并获取参数配置
    plain_register_dataset()
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        # 下载进度
        print("*******")

        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation

            img = read_image(path, format="BGR")
            start_time = time.time()
            # 获取可视化结果和预测结果
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output

                image = visualized_output.get_image()[:, :, ::-1]

                cv2.imencode(".jpg",image)[1].tofile(out_filename)
                # visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit