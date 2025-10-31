import re
import os
import json
import copy
import random
import numpy as np
import torch

from vtimellm.train.Base_dataset import BaseDataset
from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def read_from_jsonl(input_file):
    data_dict = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            key_value = json.loads(line.strip())  # 将每行 JSON 对象转换为字典
            data_dict.update(key_value)  # 更新到数据字典中
    return data_dict

class TempLocDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TempLocDataset, self).__init__(data_path, tokenizer, data_args)

        self.task_prompt = [  ## 对于 fine-tune 数据来说，是没有专门的 Task_prompt
            "During which frames can we see {}?",
            "When does {} happen in the video?",
            "At what point in the video does {} happen?",  ## 话说这里的 " " 是不是应该去掉？跟 Stage2 不符
            "When is {} depicted in the video?",
            "At what time in the video does {} take place?",
        ]
        
    def get_sources(self, i):
        pretrains = copy.deepcopy(self.list_data_dict[i])
        return self.format_TempLoc(pretrains)  ## 在这里进行所选取数据的处理
    
    def get_visual(self, sources):
            
        video = torch.zeros((100 if self.data_type == 'video' else 1, 768), dtype=torch.float16)

        video = np.load(sources['video']) # <N, 768> float16
        video = torch.from_numpy(video)
        if self.data_type == 'image' and len(video.shape) == 1: # <768>
            video = video.unsqueeze(0)
        return video
    
    def get_prompt(self, sentence, id):
        desc_prompt = random.choice(self.task_prompt)
        sentence = sentence.strip().rstrip('.')
        if len(sentence) > 1:
            sentence = sentence[0].lower() + sentence[1:]
        task_prompt = desc_prompt.format(sentence)

        RAG_prompt = 'Based on the given video, answer the question: '
        if self.use_RAG:
            tmp = id.split('_')
            video_id = '_'.join(tmp[:-2])

            if video_id in self.RAG_info:
                Base_prompt = 'The video has the following title: "{}", and description: "{}". Based on the given video, reference the title and description, answer the question: '

                RAG_title = self.RAG_info[video_id]['title']  
                RAG_description = self.RAG_info[video_id]['description']
                RAG_prompt = Base_prompt.format(RAG_title, RAG_description)

        return DEFAULT_IMAGE_TOKEN + '\n' + RAG_prompt + task_prompt

    def format_TempLoc(self, source):  

        out = {}
        out_det_target = {}
        out['video'] = '{}/{}.npy'.format(self.feature_folder, source['vid'])

        gpt_value = f"From [LOC]."
        human_value = self.get_prompt(source['query'], source['vid'])
        convers = []
        convers.append({"from": "human", "value": human_value.strip()})
        convers.append({"from": "gpt", "value": gpt_value.strip()})

        timestamps = [source['relevant_windows']]
        duration = source['duration']
        
        cur_timestamp = []
        cur_span_label_nn = []
        cur_timestamp_window = []

        for timestamp in timestamps:
            cur_t = ((torch.arange(0, self.frame_num) + 1) / self.frame_num).unsqueeze(1).repeat(1, 2)  ## 100份

            relevant_windows = torch.Tensor(timestamp)   ### 这里是 GT moment， 是 st, ed 的秒数，0~150
            num_windows = relevant_windows.shape[0]  ### GT 的数量 -- qvhl 可能会有多个 GT windowns

            relevant_windows_ts = relevant_windows / duration   ### # relevant_windows 是 st 与 ed 的秒时刻，所以被除以总长度 150 来得到归一化
            relevant_windows_ts = relevant_windows_ts.unsqueeze(0).repeat(self.frame_num, 1, 1)
            model_inputs_ts = cur_t.unsqueeze(1).repeat(1, num_windows, 1)   ### 这里两个都是 [75， num_GT_windowns， 2]

            nn_window_ts = torch.zeros_like(cur_t)
            diff_left = model_inputs_ts[..., 0]  - relevant_windows_ts[..., 0]
            diff_right = relevant_windows_ts[..., 1] - model_inputs_ts[..., 1]   ### 这里明显是在计算 落在 GT 中的片段
            assign_idx = torch.where((diff_left >= 0) * (diff_right >= 0))   ### 只取正的
            if min(assign_idx[0].shape) == 0:   # not assigned, happened in activitynet.   ### 这里在我们使用的数据集中不会出现
                nn_window_ts = relevant_windows_ts.squeeze(1)
            else:
                nn_window_ts[assign_idx[0]] = relevant_windows_ts[assign_idx[0], assign_idx[1]]   ### 这里将与 GT 相关的 clip 对应位置，都赋予了 GT moment

            cur_timestamp.append(cur_t)
            cur_span_label_nn.append(nn_window_ts)
            cur_timestamp_window.append(1 * (cur_t[:,0] >= nn_window_ts[:,0])  & (cur_t[:,1] <= nn_window_ts[:,1]))   ### 这里生成的是标志位，仅与 GT 相关的被标志为 1

        out['conversations'] = convers
        out_det_target['timestamp'] = cur_timestamp
        out_det_target['span_label_nn'] = cur_span_label_nn
        out_det_target['timestamp_window'] = cur_timestamp_window
        
        return out, out_det_target


class TempLocDataset_QVHL(TempLocDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TempLocDataset_QVHL, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):  ## 这里会被在外面的 base_dataset init 的时候用到
        self.feature_folder = os.path.join(self.data_args.feat_folder, 'qvhl_fram100_ClsPatch_feature')
        self.data_type = 'video'
        self.frame_num = 100
        if self.data_args.use_RAG:
            self.use_RAG = True
        else:
            self.use_RAG = False

    def init_list_data_dict(self):  ## 这里会被在外面的 base_dataset init 的时候用到
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'qvhighlights_train_TimeAgent.jsonl')
        self.list_data_dict = load_jsonl(data_path)  ## 具体的 json 文件 是在这里进行读取的
        if self.use_RAG:
            self.RAG_info = read_from_jsonl(self.data_args.RAG_path)


class TempLocDataset_Charades(TempLocDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TempLocDataset_Charades, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.feature_folder = os.path.join(self.data_args.feat_folder, 'charades_fram100_ClsPatch_feature')  # charades_frame100_CLIPL_feature
        self.data_type = 'video'
        self.task_prompt = ""
        self.frame_num = 100
        if self.data_args.use_RAG:
            self.use_RAG = True
        else:
            self.use_RAG = False

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'charades_sta_train_tvr_format.jsonl',)
        self.list_data_dict = load_jsonl(data_path)


class TempLocDataset_ActivityNet(TempLocDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TempLocDataset_ActivityNet, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.feature_folder = os.path.join(self.data_args.feat_folder, 'stage3_clip_feat')
        self.data_type = 'video'
        self.task_prompt = ""
        self.frame_num = 100
        if self.data_args.use_RAG:
            self.use_RAG = True
        else:
            self.use_RAG = False

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'activitynet_train_tvr_format.jsonl',)
        self.list_data_dict = load_jsonl(data_path)
