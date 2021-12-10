import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MotHeadDataset(CustomDataset):

    CLASSES = ('head', )

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)
        data_infos = [dict(
            filename=f'{i:06d}.jpg',
            width=1920,
            height=1080,
            ann=dict(
                bboxes=[],
                labels=[]
            )
        ) for i in range(1, int(ann_list[-1].split(',')[0]))]

        for i, ann_line in enumerate(ann_list):
            frame, _id, bbx, bby, bbw, bbh, ignore, *_ = ann_line.split(',')
            data_infos[int(frame) - 1]['ann']['bboxes'].append([float(bbx), float(bby), float(bbw), float(bbh)])
            data_infos[int(frame) - 1]['ann']['labels'].append(0)

        for data in data_infos:
            data['ann']['bboxes'] = np.array(data['ann']['bboxes']).astype(np.float32)
            data['ann']['labels'] = np.array(data['ann']['labels']).astype(np.int64)

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
