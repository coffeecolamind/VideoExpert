# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import re
import os
import json
import copy
import numpy as np
import torch

from vtimellm.train.Base_dataset import BaseDataset
from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


class PretrainDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(PretrainDataset, self).__init__(data_path, tokenizer, data_args)
        
    def get_sources(self, i):
        pretrains = copy.deepcopy(self.list_data_dict[i])
        return self.format_pretrains(pretrains)
    
    def get_visual(self, sources):
        """
        video = torch.zeros((100 if self.data_type == 'video' else 1, 768), dtype=torch.float16)

        video = np.load(sources['video']) # <N, 768> float16
        video = torch.from_numpy(video)
        if self.data_type == 'image' and len(video.shape) == 1: # <768>
            video = video.unsqueeze(0)
        """
        video = self.load_video(sources['video'], 100)
        return video 

    
    def format_pretrains(self, source):
        out = {}
        out_det_target = {}
        out['video'] = os.path.join(self.feature_folder, source['video_id'] )

        if '<image>' in source['conversations'][0]['value']:
            source['conversations'][0]['value'] = source['conversations'][0]['value'].replace('<image>', '<video>')
            self.data_type = 'image'

        convers, timestamps, duration = self.convert_time2loc(source)
        
        cur_timestamp = []
        cur_span_label_nn = []
        cur_timestamp_window = []

        for timestamp in timestamps:
            cur_t = ((torch.arange(0, self.frame_num) + 1) / self.frame_num).unsqueeze(1).repeat(1, 2)
            relevant_windows = torch.Tensor([timestamp])
            num_windows = relevant_windows.shape[0]
            relevant_windows_ts = relevant_windows / duration
            relevant_windows_ts = relevant_windows_ts.unsqueeze(0).repeat(self.frame_num, 1, 1)
            model_inputs_ts = cur_t.unsqueeze(1).repeat(1, num_windows, 1)

            nn_window_ts = torch.zeros_like(cur_t)
            diff_left = model_inputs_ts[..., 0]  - relevant_windows_ts[..., 0]
            diff_right = relevant_windows_ts[..., 1] - model_inputs_ts[..., 1]
            assign_idx = torch.where((diff_left >= 0) * (diff_right >= 0))
            if min(assign_idx[0].shape) == 0:
                nn_window_ts = relevant_windows_ts.squeeze(1)
            else:
                nn_window_ts[assign_idx[0]] = relevant_windows_ts[assign_idx[0], assign_idx[1]]

            cur_timestamp.append(cur_t)
            cur_span_label_nn.append(nn_window_ts)
            cur_timestamp_window.append(1 * (cur_t[:,0] >= nn_window_ts[:,0])  & (cur_t[:,1] <= nn_window_ts[:,1]))

        out['conversations'] = convers
        out_det_target['timestamp'] = cur_timestamp
        out_det_target['span_label_nn'] = cur_span_label_nn
        out_det_target['timestamp_window'] = cur_timestamp_window
        
        return out, out_det_target
    
    def convert_time2loc(self, source):
        cur_conv = []
        cur_timestamp = []
        cont_next = False
        # pattern = r'from (<s\d+>) to (<e\d+>)'
        pattern = r'(<s\d+>)\s+(?:to|and|between)\s+(<e\d+>)'

        for i, convers in enumerate(source['conversations']):
            if cont_next:
                cont_next = False
                continue

            matches = re.findall(pattern, convers['value'], flags=re.IGNORECASE)  # 使用 re.findall 找到所有匹配的 <sX> 和 <eX>

            if convers['from']=='human':
                if matches:  ## 这里用于忽略掉
                    cont_next = True
                    continue
                else:
                    cur_conv.append(convers)
            elif convers['from']=='gpt':
                convers['value'], count = re.subn(pattern, '[LOC]', convers['value'], flags=re.IGNORECASE)
                # 使用正则表达式检测并替换句子开头的 "during [LOC]" 为 "During [LOC]"
                convers['value'] = re.sub(r'(^|\.\s+|!\s+|\?\s+) \[LOC\]', r'\1During [LOC]', convers['value'])
                cur_conv.append(convers)
                
                assert len(matches) == count
                for i in matches:
                    cur_timestamp.append(i)

        if cur_conv[0]['value'][:7] != '<video>':
            cur_conv[0]['value'] = '<video>\n' + cur_conv[0]['value']

        if cur_timestamp:
            timestamp = [[source['meta']['token'][start], source['meta']['token'][end]] for start, end in cur_timestamp]
            duration = source['meta']['duration']
        else:
            timestamp = []
            duration = None

        return cur_conv, timestamp, duration

    
class PretrainDataset_2(PretrainDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(PretrainDataset_2, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.feature_folder = os.path.join(self.data_args.feat_folder, 'InternVideo', 'clipped_video')
        self.data_type = 'video'
        self.task_prompt = ""
        self.frame_num = 100

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'InternVid.json')
        self.list_data_dict = json.load(open(data_path, "r"))
        
        
class PretrainDataset_3(PretrainDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(PretrainDataset_3, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.feature_folder = os.path.join(self.data_args.feat_folder, 'ActivityNet')
        self.data_type = 'video'
        self.task_prompt = ""
        self.frame_num = 100

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'ActivityNet.json',)
        self.list_data_dict = json.load(open(data_path, "r"))
    
        
        
        