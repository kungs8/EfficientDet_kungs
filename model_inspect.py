# -*- encoding: utf-8 -*-
'''
@Time    :   2020/12/4:3:21 PM
@Author  :   yanpenggong
@Version :   1.0
@Contact :   yanpenggong@163.com
@Software:   PyCharm
@Project :   检验模型
'''

from absl import logging, app, flags
import tensorflow as tf
from typing import Text
import hparams_config

flags.DEFINE_string(name="model_name", default="efficientdet-d0", help="Model.")
flags.DEFINE_string(name="logdir", default="./logdir/", help="log路径")
flags.DEFINE_bool(name="delete_logdir", default=True, help="是否删除log路径")

flags.DEFINE_string()

FLAGS = flags.FLAGS


class ModelInspector(object):
    """用于检查模型的简单帮助程序类。"""
    def __init__(self,
                 model_name: Text,
                 logdir: Text,
                 tensorrt: Text=False,
                 use_xla: bool=False,
                 ckpt_path: Text=None,
                 export_ckpt: Text=None,
                 saved_model_dir: Text=None,
                 tflite_path: Text=None,
                 batch_size: int=1,
                 hparams: Text="",
                 **kwargs):
        self.model_name = model_name
        self.logdir = logdir
        self.tensorrt = tensorrt
        self.use_xla = use_xla
        self.ckpt_path = ckpt_path
        self.export_ckpt = export_ckpt
        self.saved_model_dir = saved_model_dir
        self.tflite_path = tflite_path

        model_config = hparams_config.get_


def mian():
    # 如果log路径存在，则删除
    if tf.io.gfile.exists(FLAGS.logdir) and FLAGS.delete_logdir:
        logging.info("Deleting log dir ...")
        tf.io.gfile.rmtree(FLAGS.logdir)

    inspector = ModelInspector()


if __name__ == '__main__':
    logging.set_verbosity(logging.WARNING)
    app.run(mian)
