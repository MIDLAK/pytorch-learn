# prepcache.py
import argparse
import sys

from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from util.logconf import logging
from dsets import LunaDataset

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class LunaPrepCacheApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=1024,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        ds = LunaDataset()
        # Сортируем кандидатов по series_uid, чтобы каждый воркер
        # обрабатывал подряд кандидатов из одного КТ-скана и
        # @functools.lru_cache(1) на getCt действительно работал.
        ds.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))

        self.prep_dl = DataLoader(
            ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass


if __name__ == '__main__':
    LunaPrepCacheApp().main()