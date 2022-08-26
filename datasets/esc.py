# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from datasets.video_db import VideoDataset

DATA_PATH = '/mnt/Zhentaob/datasets/ESC50/data'
ANNO_PATH = '/mnt/Zhentaob/datasets/ESC50/escTrainTestlist'


class ESC(VideoDataset):
    def __init__(self, subset,
                 return_video=False,
                 return_audio=True,
                 audio_clip_duration=1.,
                 audio_fps=None,
                 audio_fps_out=64,
                 audio_transform=None,
                 return_labels=False,
                 return_index=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=20,
                 sample_grad_cam_audio=False
                 ):
        assert return_video is False
        self.name = 'UCF-101'
        self.root = DATA_PATH
        self.subset = subset

        classes_fn = f'{ANNO_PATH}/classInd.txt'
        self.classes = [l.strip().split()[1] for l in open(classes_fn)]

        filenames = [ln.strip().split()[0] for ln in open(f'{ANNO_PATH}/{subset}.txt')]
        labels = [fn.split('/')[0] for fn in filenames]
        labels = [self.classes.index(cls) for cls in labels]

        self.num_classes = len(self.classes)
        self.num_videos = len(filenames)

        super(ESC, self).__init__(
            return_video=return_video,
            return_audio=return_audio,
            audio_root=DATA_PATH,
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
            sample_grad_cam_audio=sample_grad_cam_audio
        )
