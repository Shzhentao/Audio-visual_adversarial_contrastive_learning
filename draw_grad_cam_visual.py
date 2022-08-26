from PIL import Image
import torch
import torchvision.transforms as transforms
import argparse
import time
import psutil
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('--index', default='2')

args = parser.parse_args()
gpu_index = 0
# data_address = './data/ucf'

# data_output = './data/kinetics/visual_fig'
fig_output = ['./data/ucf/visual_fig_heatmap', './data/ucf/visual_mix_fig', './data/ucf/visual_origin_fig',
            './data/ucf/visual_fig_heatmap_scratch', './data/ucf/visual_mix_fig_scratch']
# fig_output_heatmap = './data/ucf/visual_fig_heatmap'
# fig_output = './data/ucf/visual_fig'
# fig_num = 4960
fig_num = 16340
for index_fig in range(len(fig_output)):
    if not os.path.exists(fig_output[index_fig]):
        os.makedirs(fig_output[index_fig])
    else:
        shutil.rmtree(fig_output[index_fig])
        os.makedirs(fig_output[index_fig])

# data_address = './data/kinetics/visual'
data_address = './data/ucf/visual'
data_address_heatmap = './data/ucf/visual/heatmap'
data_address_scratch = './data/ucf/visual_scratch'
data_address_heatmap_scratch = './data/ucf/visual_scratch/heatmap_scratch'
heapmap_list = os.listdir(data_address_heatmap)
index_heatmap = 100
# index_heatmap_max = 1000
for index_heatmap in range(len(heapmap_list)):
    heatmap = torch.load("{}/{}".format(data_address_heatmap, heapmap_list[index_heatmap]))
    heatmap_one = os.path.splitext(heapmap_list[index_heatmap])[0]
    gpu_index = int(heatmap_one[-13:-11])
    fig_index = int(heatmap_one[-10:-3])
    target_index = int(heatmap_one[-2:])
    video_origin = torch.load("{}/video-origin-{:02d}-{:07d}.pt".format(data_address, gpu_index, fig_index))
    video = torch.load("{}/video{:02d}-{:07d}.pt".format(data_address, gpu_index, fig_index))
    activations = torch.load("{}/activations-{:02d}-{:07d}.pt".format(data_address, gpu_index, fig_index))
    activations2 = torch.load("{}/activations-{:02d}-{:07d}.pt".format(data_address_scratch, gpu_index, fig_index))
    heatmap2 = torch.load("{}/video-heatmap-{:02d}-{:07d}-{:02d}.pt".format(data_address_heatmap_scratch, gpu_index, fig_index, target_index))
    # batch_index = int(args.index)
    # batch_size = video.size()[0]
    batch_size = 1

    # print(type(heatmap.detach().cpu().numpy()))

    # # print(batch_size)
    # for batch_index in range(batch_size):
    #     # video = torch.rand(4, 3, 2, 2, 2)
    #     # print(video.size())  # torch.Size([32, 3, 8, 224, 224])
    #     pc_origin = video_origin[batch_index, :, 0]
    #     pc = video[batch_index, :, 0]
    #     # print(pc.dtype)
    #     # print(pc.size())
    #     img_pc_origin = transforms.ToPILImage()(pc_origin)
    #     img_pc = transforms.ToPILImage()(pc)
    #     img_pc.show()
    #     img_pc_origin.show()
    #     # time.sleep(5)
    #     b = input()
    #     if b == 'g':
    #         for proc in psutil.process_iter():
    #             if proc.name() == 'display':
    #                 proc.kill()
    #         break
    #     else:
    #         for proc in psutil.process_iter():
    #             if proc.name() == 'display':
    #                 proc.kill()
    heatmaps = []
    index_grad = 0
    for index_grad in range(8):
        pooled_gradients = torch.mean(heatmap[0, :, index_grad].unsqueeze(0), dim=[0, 2, 3])
        activations_handle = activations[:, :, index_grad, :, :]
        for i in range(64):
            activations_handle[0, i, :, :] *= pooled_gradients[i]
        heatmap_after = torch.mean(activations_handle[0].unsqueeze(0), dim=1).squeeze().unsqueeze(0)
        heatmaps.append(heatmap_after)
    heatmap_after_all = torch.cat(heatmaps, dim=0)
    heatmap_after_all = torch.mean(heatmap_after_all, dim=0)
    # heatmap_after_all = heatmaps[0].squeeze()
    heatmaps2 = []
    index_grad2 = 0
    for index_grad2 in range(8):
        pooled_gradients2 = torch.mean(heatmap2[0, :, index_grad2].unsqueeze(0), dim=[0, 2, 3])
        activations_handle2 = activations2[:, :, index_grad2, :, :]
        for i in range(64):
            activations_handle2[0, i, :, :] *= pooled_gradients2[i]
        heatmap_after2 = torch.mean(activations_handle2[0].unsqueeze(0), dim=1).squeeze().unsqueeze(0)
        heatmaps2.append(heatmap_after2)
    heatmap_after_all2 = torch.cat(heatmaps2, dim=0)
    heatmap_after_all2 = torch.mean(heatmap_after_all2, dim=0)
    # heatmap_after_all = heatmaps[0].squeeze()
    try:
        heatmap_after_all = heatmap_after_all.detach().cpu().numpy()
        heatmap_after_all = np.maximum(heatmap_after_all, 0)
        heatmap_after_all /= np.max(heatmap_after_all)
        heatmap_after_all2 = heatmap_after_all2.detach().cpu().numpy()
        heatmap_after_all2 = np.maximum(heatmap_after_all2, 0)
        heatmap_after_all2 /= np.max(heatmap_after_all2)
        # plt.subplot(2,1,1)
        plt.matshow(heatmap_after_all)
        # plt.show()
        plt.axis('off')
        plt.savefig('{}/origin-map-{:02d}-{:07d}.jpg'.format(fig_output[0], gpu_index, fig_index))
        plt.close()
        # plt.close()
        # plt.subplot(2,1,2)
        plt.matshow(heatmap_after_all2)
        plt.axis('off')
        plt.savefig('{}/origin-map-{:02d}-{:07d}.jpg'.format(fig_output[3], gpu_index, fig_index))
        plt.close()

        # # names = ['ele.jpeg', 'shark.jpeg']
        # # img2 = cv2.imread('./data/Elephant/data/{}'.format(names[index]))
        # # print(heatmap.numpy().shape)
        # # print(img2.shape)
        # # cv2.imshow('image', heatmap)
        # # cv2.waitKey(0)

        video_one = video_origin[0, :, 0].detach().cpu().numpy()
        video_one = np.transpose(video_one, (1, 2, 0))
        # video_one2 = video[0, :, 0].detach().cpu().numpy()
        # video_one2 = np.transpose(video_one2, (1, 2, 0))
        # # print(video_one.shape)
        # heatmap_after = cv2.resize(heatmap_after_all, (video_one.shape[1], video_one.shape[0]))
        # heatmap_after = np.uint8(255 * heatmap_after)
        # heatmap_after = cv2.applyColorMap(heatmap_after, cv2.COLORMAP_JET)
        # # # print(heatmap)
        # superimposed_img = video_one + heatmap_after * 0.005
        # cv2.imwrite('{}/map-{:02d}-{:07d}.jpg'.format(fig_output[1], gpu_index, fig_index),
        #             np.uint8(superimposed_img/np.amax(superimposed_img)*255))
        origin_img = video_one
        cv2.imwrite('{}/origin-{:02d}-{:07d}.jpg'.format(fig_output[2], gpu_index, fig_index),
                    np.uint8(origin_img / np.amax(origin_img) * 255))
        # cv2.imshow('image', np.uint8(superimposed_img/np.amax(superimposed_img)*255))
        # cv2.waitKey(0)
        # cv2.imshow('image2', np.uint8(video_one/np.amax(video_one)*255))
        # cv2.imshow('image3', np.uint8(video_one2/np.amax(video_one2)*255))
        # cv2.waitKey(0)

        # # a = torch.rand(3,8,224,224).size()
        # # b = torch.rand(4,4).size()
        # # print(a == (3,8,224,225))
    except Exception:
        pass

