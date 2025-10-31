import os
import copy
import json
import random
import torch

from vtimellm.train.Base_dataset import BaseDataset
from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

ANS_MAPPING = {0:'A',1:'B',2:'C',3:'D',4:'E'}

class VidQADataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset, self).__init__(data_path, tokenizer, data_args)

        self.time_prompt = [  ## 对于 fine-tune 数据来说，是没有专门的 Task_prompt
                "During which frames can we see the Question happen?",
                "When does the Question happen in the video?",
                # "At what point in the video does the Question happen?",  ## 话说这里的 " " 是不是应该去掉？跟 Stage2 不符
                # "When is the Question depicted in the video?",
                "At what time in the video does the Question take place?",
            ]
        
    def get_sources(self, i):
        vqas = copy.deepcopy(self.list_data_dict[i])
        return self.format_vqas(vqas)  ## 在这里进行所选取数据的处理
    
    def get_visual(self, sources):
        
        video = self.load_video(sources['video'], 100)
        return video  ## frames [100, 3, 224, 224]
        
    def get_prompt_human_gpt(self, source):

        if not source['question'].endswith('?'):
                source['question'] += '?'

        question_prompt = 'Question: ' + source['question']  ## 这里是构造的输入的 Text ，格式为 Question: ... ? Option A: ... . Option B: ... .

        for j in range(source['num_option']):
            a = source['a{}'.format(j)]
            question_prompt += ' Option {}: '.format(ANS_MAPPING[j])
            question_prompt += a
            
        answers_prompt = 'Option ' + ANS_MAPPING[int(source['answer'])] + '.'  ## 这里是答案 ，形式为 Option E
        
        if 'timestamps' in source:
            question_time_prompt = random.choice(self.time_prompt)
            answer_time_prompt  = f"From [LOC]."

            question_prompt = question_prompt + ' ' + question_time_prompt + ' ' + 'Then, c'
            question_prompt  =  question_prompt + self.task_prompt[1:]

            answers_prompt = answer_time_prompt + ' ' + answers_prompt

        else:
            question_prompt  =  question_prompt + ' ' + self.task_prompt

        return DEFAULT_IMAGE_TOKEN + '\n' + question_prompt, answers_prompt
    
    def format_vqas(self, source):
        out = {}
        out_det_target = {}
        out['video'] = os.path.join(self.feature_folder, (source['video']+'.mp4'))

        human_value, gpt_value = self.get_prompt_human_gpt(source)
        
        convers = []
        convers.append({"from": "human", "value": human_value.strip()})
        convers.append({"from": "gpt", "value": gpt_value.strip()})

        if 'timestamps' in source:
            timestamps = [source['timestamps']]
            duration = source['duration']
        else:
            timestamps = []
            duration = None

        cur_timestamp = []
        cur_span_label_nn = []
        cur_timestamp_window = []

        for timestamp in timestamps:  ## 经过 for 之后是 [[0.1, 9.3]]
            cur_t = ((torch.arange(0, self.frame_num) + 1) / self.frame_num).unsqueeze(1).repeat(1, 2)  ## 100份

            relevant_windows = torch.Tensor(timestamp)   ### 这里是 GT moment， 是 st, ed 的秒数，0~150  ## 这里结束后是 [1, 2]
            num_windows = relevant_windows.shape[0]

            relevant_windows_ts = relevant_windows / duration   ### # relevant_windows 是 st 与 ed 的秒时刻，所以被除以总长度 150 来得到归一化
            relevant_windows_ts = relevant_windows_ts.unsqueeze(0).repeat(self.frame_num, 1, 1)  ## 这里结束后是 -[100,1,2]
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

     
class VidQADataset_Nextqa(VidQADataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset_Nextqa, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):  ## 这里会被在外面的 base_dataset init 的时候用到
        self.feature_folder = os.path.join(self.data_args.feat_folder, 'NExT_Videos')  # NExT_Videos
        self.data_type = 'video'
        self.frame_num = 100
        self.task_prompt = "Considering the information presented in the video, select the correct answer from the options (A, B, C, D, E)." # "Answer the question using a short phrase."
        
    def init_list_data_dict(self):  ## 这里会被在外面的 base_dataset init 的时候用到
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'Next_QA_Train.json')
        self.list_data_dict = json.load(open(data_path, "r"))  ## 具体的 json 文件 是在这里进行读取的


class VidQADataset_msvdqa(VidQADataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset_msvdqa, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'msvdqa', 'YouTubeClips')
        self.visual_data_type = 'video'
        self.task_prompt = "Answer the question using a single word or phrase."

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'msvdqa', 'train_processed.json')
        self.list_data_dict = json.load(open(data_path, "r"))
        
        
class VidQADataset_msrvttqa(VidQADataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset_msrvttqa, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'msrvttqa', 'TrainValVideo')
        self.visual_data_type = 'video'
        self.task_prompt = "Answer the question using a single word or phrase."

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'msrvttqa', 'train_processed.json')
        self.list_data_dict = json.load(open(data_path, "r"))


class VidQADataset_videochat(VidQADataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset_videochat, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'videochat_instruct_11k', 'videos')
        self.visual_data_type = 'video'
        self.task_prompt = ""

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'videochat_instruct_11k', 'videochat_instruct_11k.json')
        self.list_data_dict = json.load(open(data_path, "r"))
    
        