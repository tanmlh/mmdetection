# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from functools import partial
from pathlib import Path
import pdb
import os
from tqdm import tqdm

import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Sequence, Union

from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmdet.registry import MODELS
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils.planet_inferencer import InferencePipeline

try:
    from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def load_checkpoint(model,
                    filename: str,
                    map_location: Union[str, Callable] = 'cpu',
                    strict: bool = False,
                    revise_keys: list = [(r'^module.', '')]):
    """Load checkpoint from given ``filename``.

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str or callable): A string or a callable function to
            specifying how to remap storage locations.
            Defaults to 'cpu'.
        strict (bool): strict (bool): Whether to allow different params for
            the model and checkpoint.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].
    """
    checkpoint = _load_checkpoint(filename, map_location=map_location)
    checkpoint = _load_checkpoint_to_model(
        model, checkpoint, strict, revise_keys=revise_keys)

    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Get a detector flops')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=1e9,
        help='num images of calculate model flops')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference(args, logger):
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'Some config files, such as configs/yolact and configs/detectors,'
            'may have compatibility issues with torch.jit when torch<1.12. '
            'If you want to calculate flops for these models, '
            'please make sure your pytorch version is >=1.12.')

    config_name = Path(args.config)
    if not config_name.exists():
        logger.error(f'{config_name} not found.')

    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name
    save_cfg = cfg.get('save_cfg', {})

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    result = {}
    data_loader = Runner.build_dataloader(cfg.test_dataloader)
    model = MODELS.build(cfg.model)
    if 'load_from' in cfg:
        load_checkpoint(model, cfg.load_from)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    pipeline = InferencePipeline(
        model=model,
        num_images=len(data_loader),
        save_cfg=save_cfg,
        cpu_workers=2
    )
    results = pipeline.run(data_loader)

    """
    for idx, data_batch in enumerate(tqdm(data_loader, desc='inferencing...')):
        if idx == args.num_images:
            break
        data = model.data_preprocessor(data_batch)
        with torch.no_grad():
            # results = model.predict(data['inputs'], data['data_samples'])
            imgs = data['inputs']
            batch_data_samples = data['data_samples']

            results = model.predict_sem_seg(imgs, batch_data_samples) # GPU
            results = model.predict_mosaic_sem_seg(imgs, batch_data_samples) # CPU
            results = model.seg_poly_head.predict_seg2ins(imgs, results) # CPU
            results = model.seg_poly_head.poly_head.predict_sample_segments(imgs, results) # CPU
            results = model.seg_poly_head.poly_head.predict_gcp(imgs, results) # GPU
            results = model.seg_poly_head.poly_head.predict_assemble_segments(imgs, results) # CPU
            results = model.seg_poly_head.poly_head.predict_dp(imgs, results) # GPU

        if save_cfg.get('save_results', False):
            poly_jsons = results[0].pred_instances['segmentations']
            file_name = results[0].metainfo['img_path'].split('/')[-1].split('.')[0]
            out_dir = save_cfg['out_dir']
            out_path = os.path.join(out_dir, file_name + '.geojson')
            out_scale = save_cfg.get('out_poly_scale', 1.0)

            os.makedirs(out_dir, exist_ok=True)

            transform = results[0].metainfo['tif_meta']['transform']
            crs = results[0].metainfo['tif_meta']['crs']
            polygon_utils.save_polygons(poly_jsons, transform, crs, out_path, out_scale)

    """

    del data_loader

    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    result = inference(args, logger)

if __name__ == '__main__':
    main()
