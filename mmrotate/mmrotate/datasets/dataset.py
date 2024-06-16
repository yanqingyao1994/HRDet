import os.path as osp
from mmrotate.registry import DATASETS
from mmrotate.datasets import DOTADataset

@DATASETS.register_module()
class DOTAMaskDataset(DOTADataset):
    def load_data_list(self):
        data_list = super().load_data_list()
        if self.ann_file:
            for data_info in data_list:
                file_name = data_info['file_name']
                img_path = self.data_prefix['img_path']
                mask_path = self.data_prefix['mask_path']
                data_info['seg_map_path'] = osp.join(img_path, mask_path, file_name)
        return data_list
