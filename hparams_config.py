# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/4:5:47 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   用于模型架构和训练的Hparams
'''
import ast
import collections
import copy
from typing import Text, Dict, Any

import six
import yaml
import tensorflow as tf


def eval_str_fn(val):
    if val in {"true", "false"}:
        return val == "true"
    try:
        return ast.literal_eval(val)  # string <=> dict
    except (ValueError, SyntaxError):
        return val


class Config(object):
    """配置实用程序类"""
    def __init__(self, config_dict=None):
        self.update(config_dict)

    def __setattr__(self, key, value):
        """设置类实例属性 如Config().name='tom'，自调用__setattr__"""
        self.__dict__[key] = Config(value) if isinstance(value, dict) else copy.deepcopy(value)

    def __getattr__(self, item):
        """内置使用点号获取实例属性属性如 s.name，自调用__getattr__"""
        return self.__dict__[item]

    def __getitem__(self, item):
        """使用[]获取实例属性 如Config()['name']，自调用__getitem__"""
        return self.__dict__[item]

    def __repr__(self):
        """类的实例化对象用来做“自我介绍”"""
        return repr(self.as_dict())

    def __deepcopy__(self, memodict):
        return type(self)(self.as_dict())

    def __str__(self):
        try:
            return yaml.dump(data=self.as_dict(), indent=4)  # 序列化字符串的格式
        except TypeError:
            return str(self.as_dict())

    def _update(self, config_dict, allow_new_keys=True):
        """
        递归更新内部参数
        :param config_dict:
        :param all_new_keys:
        :return:
        """
        if not config_dict:
            return
        for key, value in six.__dict__:
            if key not in self.__dict__:
                if allow_new_keys:
                    self.__setattr__(key=key, value=value)
                else:
                    raise KeyError("Key '{}' does not exist for overriding.".format(key))
            else:
                if isinstance(self.__dict__[key], Config) and isinstance(value, dict):
                    self.__dict__[key]._update(value, allow_new_keys)
                elif isinstance(self.__dict__[key], Config) and isinstance(value, Config):
                    self.__dict__[key]._update(value.as_dict(), allow_new_keys)
                else:
                    self.__setattr__(key, value)

    def get(self, key, default_value=None):
        return self.__dict__.get(key, default_value)

    def update(self, config_dict):
        """在允许新keys的同时更新参数"""
        self._update(config_dict=config_dict, allow_new_keys=True)

    def keys(self):
        return self.__dict__.keys()

    def override(self, config_dict_or_str, allow_new_keys=False):
        """
        在不允许新keys的同时更新参数
        :param config_dict_or_str:
        :param allow_new_keys:
        :return:
        """
        if isinstance(config_dict_or_str, str):
            if not config_dict_or_str:
                return
            elif "=" in config_dict_or_str:
                config_dict = self.parse_from_str(config_dict_or_str)
            elif config_dict_or_str.endswith(".yaml"):
                config_dict = self.parse_from_yaml(config_dict_or_str)
            else:
                raise ValueError("Invalid string {}, must end with .yaml or contains '='.".format(config_dict_or_str))
        elif isinstance(config_dict_or_str, dict):
            config_dict = config_dict_or_str
        else:
            raise ValueError("Unknown value type:{}".format(config_dict_or_str))
        self._update(config_dict, allow_new_keys)

    def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
        """解析yaml文件并返回字典"""
        with tf.io.gfile.GFile(name=yaml_file_path, mode="r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict

    def save_to_yaml(self, yaml_file_path):
        """将字典写到yaml文件中"""
        with tf.io.gfile.GFile(name=yaml_file_path, mode="w") as f:
            yaml.dump(self.as_dict(), f, default_flow_style=False)

    def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
        """
        将类似“ x.y = 1，x.z = 2”的字符串解析为嵌套的字典{x：{y：1，z：2}}。
        :param config_str:
        :return:
        """
        if not config_str:
            return {}
        config_str = {}
        try:
            for kv_pair in config_str.split(","):
                if not kv_pair:
                    continue
                key_str, value_str = kv_pair.split("=")
                key_str = key_str  # 去除首尾空格

                def add_kv_recursive(key, value):
                    """将x.y.z = tt递归解析为{x：{y：{z：tt}}}"""
                    if "." not in key:
                        if "*" in value:
                            # 我们以 '*' 拆分数组
                            return {key: [eval_str_fn(val) for val in value.split("*")]}
                        return {key: eval_str_fn(value)}
                    pos = key.index(".")
                    return {key[:pos]: add_kv_recursive(key[pos+1:], value)}

                def merge_dict_recursive(target, src):
                    """递归合并两个嵌套字典"""
                    for key in src.keys():
                        if ((key in target and isinstance(target[key], dict) and isinstance(src[key], collections.abc.Mapping))):
                            merge_dict_recursive(target=target[key], src=src[key])
                        else:
                            target[key] = src[key]
                merge_dict_recursive(target=config_str, src=add_kv_recursive(key=key_str, value=value_str))
        except ValueError:
            raise ValueError("Invalid config_str:{}".format(config_str))

    def as_dict(self):
        """返回字典表示"""
        config_dict = {}
        for key, value in six.iteritems(self.__dict__):
            if isinstance(value, Config):
                config_dict[key] = value.as_dict()
            else:
                config_dict[key] = copy.deepcopy(value)
        return config_dict


def default_detection_configs():
    """
    默认的参数
    :return: 返回默认的检测配置
    """
    h = Config()

    # 模型的名称
    h.name = "efficientdet-d1"

    # 激活类型：请参见utils.py中的activation_fn。
    h.act_type = "swish"

    # 输入预处理参数
    h.image_size = 640  # 整数或字符串WxH，例如640x320。
    h.target_size = None
    h.input_rand_hflip = True
    h.jitter_min = 0.1
    h.jitter_max = 2.0
    h.autoaugment_policy = None
    h.grid_mask = False
    h.sample_image = None
    h.map_freq = 5  # AP评估频率，以epochs为单位

    # 数据集特定参数
    # 对于COCO，更新为91；对于pascal，更新为21
    h.num_classes = 90  # 1+实际类，0保留作为background
    h.seg_num_classes = 3  # 细分类别
    h.heads = ["object_detection"]  # "object_detection", "segmentation"

    h.skip_crowd_during_training = True
    h.label_map = None  # “ coco”，“ voc”，“ waymo”的字典或字符串
    h.max_instances_per_image = 100  # COCO的默认值为100
    h.regenerate_source_id = False

    # 模型架构
    h.min_level = 3
    h.max_level = 7
    h.num_scales = 3
    # 比率w / h：2.0表示w ＝ 1.4，h ＝ 0.7。可以使用每个数据集的k均值进行计算
    h.aspect_ratios = [1.0, 2.0, 0.5]
    h.anchor_scale = 4.0
    # batchnorm训练模式
    h.is_training_bn = True
    # 优化
    h.momentum= 0.9
    h.optimizer = "sgd"  # 可以是 "adam"/"sgd"
    h.learning_rate = 0.08  # 对"adam"是 0.08
    h.lr_warmup_init = 0.008  # 对"adam"是 0.008
    h.lr_warmup_epoch = 1.0
    h.first_lr_drop_epoch = 200.0
    h.second_lr_drop_epoch = 250.0
    h.poly_lr_power = 0.9
    h.clip_gradients_norm = 10.0
    h.num_epochs = 300
    h.data_format = "channels_last"

    # 分类损失
    h.label_smoothing = 0.0  # 0.1是一个很好的默认值
    # 重要的损失参数
    h.alpha = 0.25
    h.gamma = 1.5

    # 本地化损失
    h.delta = 0.1  # Huber损失的正则化参数
    # 总损失= box_loss * box_loss_weight + iou_loss * iou_loss_weight
    h.box_loss_weight = 50.0
    h.iou_loss_type = None
    h.iou_loss_weight = 1.0

    # L2正则损失
    h.weight_decay = 4e-5
    h.strategy = None  # "tpu", "gpus", None
    h.mixed_precision = False  # 若为False，则使用float32
    h.model_optimizations = {}  # 修剪

    # 用于检测
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_num_filters = 88
    h.separable_conv = True
    h.apply_bn_for_resampling = True
    h.conv_after_downsample = False
    h.conv_bn_act_pattern = False
    h.drop_remainder = True  # 最后一批评估的剩余数drop

    # 对于后处理nms，必须是字典
    h.nms_configs = {
        "method": "gaussian",
        "iou_thresh": None,  # 使用基于方法的默认值
        "score_thresh": 0,
        "sigma": None,
        "pyfunc": False,
        "max_nms_inputs": 0,
        "max_output_size": 100,
    }

    # version
    h.fpn_name = None
    h.fpn_weight_method = None
    h.fpn_config = None

    # 默认情况下没有随机depth
    h.survival_prob = None
    h.img_summary_steps = None
    h.lr_decay_method = "cosine"
    h.moving_average_decay = 0.9998
    h.ckpt_var_scope = None  # ckpt变量范围
    # 如果为true，则在形状不匹配时跳过加载预训练的权重
    h.skip_mismatch = True

    h.backbone_name = "efficientnet-b1"
    h.backbone_config = None
    h.var_freeze_expr = None

    # 在传统模型和keras模型之间切换的临时标志
    h.use_keras_model = True
    h.dataset_type = None
    h.positives_momentum = None
    h.grad_checkpoint = False
    return h


efficientdet_model_param_dict = {
    "efficientdet-d0":
        dict(
            name="efficientdet-d0",
            backbone_name="efficientnet-b0",
            image_size=512,
            fpn_num_filters=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
        ),
    "efficientdet-d1":
        dict(
            name="efficientdet-d1",
            backbone_name="efficientnet-b1",
            image_size=640,
            fpn_num_filters=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
        ),
    "efficientdet-d2":
        dict(
            name="efficientdet-d2",
            backbone_name="efficientnet-b2",
            image_size=768,
            fpn_num_filters=112,
            fpn_cell_repeats=3,
            box_class_repeats=3,
        ),
    "efficientdet-d3":
        dict(
            name="efficientdet-d3",
            backbone_name="efficientnet-b3",
            image_size=896,
            fpn_num_filters=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
        ),
    "efficientdet-d4":
        dict(
            name="efficientdet-d4",
            backbone_name="efficientnet-b4",
            image_size=1024,
            fpn_num_filters=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    "efficientdet-d5":
        dict(
            name="efficientdet-d5",
            backbone_name="efficientnet-b0",
            image_size=1280,
            fpn_num_filters=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    "efficientdet-d6":
        dict(
            name="efficientdet-d6",
            backbone_name="efficientnet-b6",
            image_size=1280,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            fppn_weight_method="sum",  # 使用未加权的总和来保持稳定性
        ),
    "efficientdet-d7":
        dict(
            name="efficientdet-d7",
            backbone_name="efficientnet-b6",
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=5.0,
            fppn_weight_method="sum",  # 使用未加权的总和来保持稳定性
        ),
    "efficientdet-d7x":
        dict(
            name="efficientdet-d7x",
            backbone_name="efficientnet-b7",
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=4.0,
            max_level=8,
            fppn_weight_method="sum",  # 使用未加权的总和来保持稳定性
        ),
}

efficientdet_lite_model_param_dict = {
    "efficientdet-lite0":
        dict(
            name="efficientdet-lite0",
            backbone_name="efficientnet-lite0",
            image_size=320,
            fpn_num_filters=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
            act_type="relu",
            fpn_weight_method="sum",
            anchor_scale=3.0,
        ),
    "efficientdet-lite1":
        dict(
            name="efficientdet-lite1",
            backbone_name="efficientnet-lite1",
            image_size=384,
            fpn_num_filters=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
            act_type="relu",
            fpn_weight_method="sum",
            anchor_scale=3.0,
        ),
    "efficientdet-lite2":
        dict(
            name="efficientdet-lite2",
            backbone_name="efficientnet-lite2",
            image_size=448,
            fpn_num_filters=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
            act_type="relu",
            fpn_weight_method="sum",
            anchor_scale=3.0,
        ),
    "efficientdet-lite3":
        dict(
            name="efficientdet-lite3",
            backbone_name="efficientnet-lite3",
            image_size=512,
            fpn_num_filters=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
            act_type="relu",
            fpn_weight_method="sum",
        ),
    "efficientdet-lite4":
        dict(
            name="efficientdet-lite4",
            backbone_name="efficientnet-lite4",
            image_size=512,
            fpn_num_filters=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
            act_type="relu",
            fpn_weight_method="sum",
        ),
}


def get_efficientdet_config(model_name):
    """
    根据模型名称获取EfficientDet的默认配置。
    :param model_name: 模型名称
    :return:
    """
    h = default_detection_configs()
    if model_name in efficientdet_model_param_dict:
        h.override(efficientdet_model_param_dict[model_name])
    elif model_name in efficientdet_lite_model_param_dict:
        h.override(efficientdet_lite_model_param_dict[model_name])
    else:
        raise ValueError("Unknown model name:{}".format(model_name))
    return h


def get_detection_config(model_name):
    """
    检测模型名称对应的模型参数
    :param model_name: 模型名称
    :return:
    """
    if model_name.startswith("efficientdet"):
        return get_efficientdet_config(model_name)
    else:
        raise ValueError("Model name must start with efficientdet.")