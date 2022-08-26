# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import glob
import numpy as np

DATA_PATH = '/mnt/Zhentaoc/datasets/UCF101_VS'  # DATA_PATH = '/mnt/Zhentaob/data/datasets/kinetics_small'

from datasets.video_db import VideoDataset


class UCF2(VideoDataset):
    def __init__(self, subset,
                 return_video=True,
                 video_clip_duration=1.,
                 video_fps=25.,
                 video_transform=None,
                 return_audio=False,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=1,
                 ):
        classes = sorted(os.listdir(f"{DATA_PATH}/{subset}"))
        filenames = ['/'.join(fn.split('/')[-2:]) for fn in glob.glob(f"{DATA_PATH}/{subset}/*/*.avi")]
        labels = [classes.index(fn.split('/')[-2]) for fn in filenames]

        super(UCF2, self).__init__(
            return_video=return_video,
            video_root=f"{DATA_PATH}/{subset}",
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root=f"{DATA_PATH}/{subset}",
            audio_fns=filenames,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            max_offsync_augm=max_offsync_augm,
            mode=mode,
            clips_per_video=clips_per_video,
        )

        self.name = 'Kinetics dataset'
        self.root = f"{DATA_PATH}/{subset}"
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array([fn.split('/')[-1].split('.')[0].encode('utf-8') for fn in filenames])
