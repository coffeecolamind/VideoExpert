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

class DVCDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(DVCDataset, self).__init__(data_path, tokenizer, data_args)

        self.desc_prompts = [
            "Provide a detailed description of the given video.",
            "Describe the provided video in detail.",
            "Summarize the visual content of the video.",
            "Write a informative summary of the video.",
            "Provide a detailed description of the events shown in the video."
        ]
        self.time_prompts = [
            "Each sentence should begin with the timestamps.",
            "At the beginning of each sentence, include the timestamps.",
            "Prepend each sentence with its timestamps.",
            "Please specify the exact time of each sentence."
        ]
        
    def get_sources(self, i):
        pretrains = copy.deepcopy(self.list_data_dict[i])
        return self.format_Dense_Video_Captions(pretrains)  ## 在这里进行所选取数据的处理
    
    def get_visual(self, sources):

        if self.feature_source_type == 'YoucookII':
            video = torch.zeros((100 if self.data_type == 'video' else 1, 257, 768), dtype=torch.float16)
            video = np.load(sources['video']) # <N, 768> float16
            video = torch.from_numpy(video)
            if self.data_type == 'image' and len(video.shape) == 1: # <768>
                video = video.unsqueeze(0)

        elif self.feature_source_type == 'Activitynet':
            video = self.load_video(sources['video'], 100)
        return video
    
    def get_prompt_human(self, id):
        task_prompt = random.choice(self.desc_prompts) + ' ' + random.choice(self.time_prompts)

        RAG_prompt = 'Based on the given video, answer the question: '
        if self.use_RAG:
            video_id = id

            if video_id in self.RAG_info:
                Base_prompt = 'The video has the following title: "{}", and description: "{}". Based on the given video, reference the title and description, answer the question: '

                RAG_title = self.RAG_info[video_id]['title']  
                RAG_description = self.RAG_info[video_id]['description']
                RAG_prompt = Base_prompt.format(RAG_title, RAG_description)

        return DEFAULT_IMAGE_TOKEN + '\n' + RAG_prompt + task_prompt

    def get_prompt_gpt(self, source):

        for i, sentence in enumerate(source['sentences']):
            if not sentence.endswith('.'):
                sentence += '.'

            if i == 0:
                gpt = f"From [LOC], " + sentence
            elif i>=0:
                gpt +=' ' + f"From [LOC], " + sentence

        assert len(source['timestamps']) == (i+1)
        
        return gpt, source['timestamps']


    def format_Dense_Video_Captions(self, source):  

        out = {}
        out_det_target = {}
        if self.feature_source_type == 'YoucookII':
            out['video'] = '{}/{}.npy'.format(self.feature_folder, source['vid'])
        elif self.feature_source_type == 'Activitynet':
            out['video'] = os.path.join(self.feature_folder, source['video_id'])
        
        human_value = self.get_prompt_human(source['vid'])
        gpt_value, timestamps = self.get_prompt_gpt(source)
        duration = source['duration']

        convers = []
        convers.append({"from": "human", "value": human_value.strip()})
        convers.append({"from": "gpt", "value": gpt_value.strip()})
        
        cur_timestamp = []
        cur_span_label_nn = []
        cur_timestamp_window = []

        for timestamp in timestamps:
            cur_t = ((torch.arange(0, self.frame_num) + 1) / self.frame_num).unsqueeze(1).repeat(1, 2)  ## 100份

            relevant_windows = torch.Tensor([timestamp])   ### 这里是 GT moment， 是 st, ed 的秒数，0~150
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


class DVCDataset_Activitynet(DVCDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(DVCDataset_Activitynet, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):  ## 这里会被在外面的 base_dataset init 的时候用到
        self.feature_folder = os.path.join(self.data_args.feat_folder, 'ActivityNet')
        self.data_type = 'video'
        self.frame_num = 100
        self.feature_source_type = 'Activitynet'
        if self.data_args.use_RAG:
            self.use_RAG = True
        else:
            self.use_RAG = False

    def init_list_data_dict(self):  ## 这里会被在外面的 base_dataset init 的时候用到
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'ActivityNet', 'ActivityNet_DVC_captions', 'train_new.json')
        data_dict = json.load(open(data_path, "r"))  ## 具体的 json 文件 是在这里进行读取的
        for k in data_dict:
            v = data_dict[k]
            v['vid'] = k
            self.list_data_dict.append(v)

        if self.use_RAG:
            self.RAG_info = read_from_jsonl(self.data_args.RAG_path)


class DVCDataset_Youcook2(DVCDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(DVCDataset_Youcook2, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.feature_folder = os.path.join(self.data_args.feat_folder, '')
        self.data_type = 'video'
        self.frame_num = 100
        self.feature_source_type = 'YoucookII'
        if self.data_args.use_RAG:
            self.use_RAG = True
        else:
            self.use_RAG = False

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'YouCookII', 'YouCookII_train.json',)
        data_dict = json.load(open(data_path, "r"))
        for k in data_dict:
            v = data_dict[k]
            v['vid'] = k
            self.list_data_dict.append(v)

        if self.use_RAG:
            self.RAG_info = read_from_jsonl(self.data_args.RAG_path)

