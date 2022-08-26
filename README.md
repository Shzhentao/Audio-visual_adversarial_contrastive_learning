# AVAC

This repo provides a PyTorch implementation and pretrained models for AVAC, as described in our paper:

to run with modal:
```
python main-avac.py configs/main/avac/kinetics/Cross-N65536.yaml --dist-url tcp://127.0.0.1:12345 --multiprocessing-distributed --world-size 1 --rank 0 --quiet
```

To evaluate on UCF Split-1 (full finetuning), simply run:

```
python eval-action-recg.py configs/benchmark/ucf/8at16-fold1.yaml configs/main/avac/kinetics/Cross-N65536.yaml --distributed --port 12345 --quiet
```

To evaluate on hmdb51 Split-1 (full finetuning), simply run:
```
python eval-action-recg.py configs/benchmark/hmdb51/8at16-fold1.yaml configs/main/avac/kinetics/Cross-N65536.yaml --distributed --port 12345 --quiet
```

to draw grad cam visual, run:
```
python eval-action-recg.py configs/benchmark/ucf/8at16-fold1.yaml configs/main/avac/kinetics/Cross-N65536.yaml --distributed --port 12345 --quiet --sample_grad_cam
python draw_grad_cam_visual.py
```
