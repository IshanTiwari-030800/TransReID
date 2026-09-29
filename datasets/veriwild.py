import os
import os.path as osp

from .bases import BaseImageDataset


class VeriWild(BaseImageDataset):
    """
    VeRi-Wild
    Reference:
    Lou et al. VERI-Wild: A Large Dataset and a New Method for Vehicle
    Re-Identification in the Wild. CVPR 2019.

    URL: https://github.com/PKU-IMRE/VERI-Wild

    Dataset statistics:
    # train_list_start0: 30671 vehicles for model training (already 0-indexed)
    # test_3000 / test_5000 / test_10000: small/medium/large test sets
    """
    dataset_dir = 'Veri-Wild-1'

    def __init__(self, root='', verbose=True, test_size=3000, **kwargs):
        super(VeriWild, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.img_root = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.train_list = osp.join(self.split_dir, 'train_list_start0.txt')
        self.test_size = test_size

        if self.test_size not in [3000, 5000, 10000]:
            raise RuntimeError('"{}" is not available, expected one of [3000, 5000, 10000]'.format(self.test_size))

        self.gallery_list = osp.join(self.split_dir, 'test_{}_id.txt'.format(self.test_size))
        self.query_list = osp.join(self.split_dir, 'test_{}_id_query.txt'.format(self.test_size))

        self._check_before_run()
        self._build_image_index()

        train = self._process_list(self.train_list)
        query = self._process_list(self.query_list)
        gallery = self._process_list(self.gallery_list)

        if verbose:
            print("=> VeRi-Wild loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.img_root):
            raise RuntimeError("'{}' is not available".format(self.img_root))
        if not osp.exists(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))
        if not osp.exists(self.query_list):
            raise RuntimeError("'{}' is not available".format(self.query_list))
        if not osp.exists(self.gallery_list):
            raise RuntimeError("'{}' is not available".format(self.gallery_list))

    def _build_image_index(self):
        """
        Images are stored as images/<range_folder>/<vehicle_id>/<image>.jpg
        List files only give '<vehicle_id>/<image>.jpg', so map each
        vehicle_id folder to the range folder that contains it once, up front.
        """
        self.vid2dir = {}
        for range_dir in os.listdir(self.img_root):
            range_path = osp.join(self.img_root, range_dir)
            if not osp.isdir(range_path):
                continue
            for vid in os.listdir(range_path):
                self.vid2dir[vid] = osp.join(range_path, vid)

    def _process_list(self, list_path):
        dataset = []
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            rel_path, pid, camid = line.split(' ')
            pid = int(pid)
            camid = int(camid)
            vid = rel_path.split('/')[0]
            img_name = rel_path.split('/')[1]
            img_path = osp.join(self.vid2dir[vid], img_name)
            viewid = 1
            dataset.append((img_path, pid, camid, viewid))

        return dataset
