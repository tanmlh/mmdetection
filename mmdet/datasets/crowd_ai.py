# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset
from .api_wrappers import COCO
from mmengine.fileio import get_local_path
import pdb
import json

import copy
import os.path as osp
from typing import List, Union

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset





@DATASETS.register_module()
class CrowdAIDataset(CocoDataset):
    """Dataset for iSAID instance segmentation.

    iSAID: A Large-scale Dataset for Instance Segmentation
    in Aerial Images.

    For more detail, please refer to "projects/iSAID/README.md"
    """

    METAINFO = dict(
        classes=('building'),
        palette=[(0, 0, 255)])

    def __init__(self, coco_res_path=None, **kwargs):
        self.coco_res_path = coco_res_path
        super().__init__(**kwargs)



    def load_data_list(self):
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

            if self.coco_res_path is not None:
                submission_file = json.loads(open(self.coco_res_path).read())
                # coco = COCO(gt_json_path)
                coco = self.coco.loadRes(submission_file)
                self.coco = coco
                # pass

        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.getCatIds(self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        # self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        self.cat_img_map = copy.deepcopy(self.coco.catToImgs)

        img_ids = self.coco.getImgIds()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.loadImgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.getAnnIds([img_id])
            raw_ann_info = self.coco.loadAnns(ann_ids)

            if self.coco_res_path is not None:
                for x in raw_ann_info:
                    x['segmentation'] = x['polygon']

            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        data_list = [data_list[x] for x in [1416, 1426, 1682, 1379, 751]]
        return data_list
