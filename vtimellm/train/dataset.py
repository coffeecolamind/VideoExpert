import random
import copy
import json
import torch
import transformers
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from vtimellm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from vtimellm.datasets.pretrain_dataset import PretrainDataset_2, PretrainDataset_3
from vtimellm.datasets.temploc_dataset import TempLocDataset_QVHL, TempLocDataset_Charades, TempLocDataset_ActivityNet
from vtimellm.datasets.dvc_dataset import DVCDataset_Youcook2, DVCDataset_Activitynet
from vtimellm.datasets.vidqa_dataset import VidQADataset_Nextqa

from vtimellm.arguments import DataArguments

from vtimellm.train.Base_dataset import preprocess

class HybridDataset(Dataset):
    """Dataset for HybridDataset supervised pre/fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,):
        super(HybridDataset, self).__init__()

        self.samples_per_epoch = data_args.samples_per_epoch
        self.tasks = data_args.tasks.split("||")
        task_sample_rate = data_args.task_sample_rate
        s = sum(task_sample_rate)
        self.task_sample_rate = [float(x)/s for x in task_sample_rate]
        assert len(self.task_sample_rate) == len(self.tasks)

        ds_dict = {
            'pretrain':{
                'Stage2': PretrainDataset_2,
                'Stage3': PretrainDataset_3,
            },
            'temp_loc':{
                'QVHL': TempLocDataset_QVHL,
                'Charades': TempLocDataset_Charades,
                'ActivityNet': TempLocDataset_ActivityNet
            },
            'dvc':{
                'Youcook2': DVCDataset_Youcook2,
                'ActivityNet': DVCDataset_Activitynet
            },
            'videoqa':{
                'TVQA': None ,
                'NextQA': VidQADataset_Nextqa ,
            }
        }

        self.all_datasets = []
        self.all_sample_rate = []

        for task in self.tasks:
            task_data = getattr(data_args, task + '_data', '')
            datasets = []
            sample_counts = []

            for data in task_data.split('||'):
                dataset = ds_dict[task][data](data_path, tokenizer, data_args)
                datasets.append(dataset)
                sample_counts.append(len(dataset))

            sample_rate = getattr(data_args, task + '_sample_rate', sample_counts)
            assert len(sample_rate) == len(datasets)
            s = sum(sample_rate)
            sample_rate = [float(x)/s for x in sample_rate]
            self.all_sample_rate.append(sample_rate)
            self.all_datasets.append(datasets)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        rng = np.random.RandomState()
        task = rng.choice(list(range(len(self.all_datasets))), p=self.task_sample_rate)
        dataset = rng.choice(list(range(len(self.all_datasets[task]))), p=self.all_sample_rate[task])
        return self.all_datasets[task][dataset][0]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = json.load(open(data_path, "r"))
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(self.list_data_dict[i])

        data_type = 'video'
        if '<image>' in source['conversations'][0]['value']:
            source['conversations'][0]['value'] = source['conversations'][0]['value'].replace('<image>', '<video>')
            data_type = 'image'

        if 'meta' in source:
            def convert(duration, x):
                x = x / duration * 100
                x = str(min(round(x), 99))
                if len(x) == 1:
                    x = "0" + x
                return x

            replace_set = []
            for k, v in source['meta']['token'].items():
                replace_set.append((k, convert(source['meta']['duration'], v)))
            for l in range(len(source['conversations'])):
                for x1, x2 in replace_set:
                    source['conversations'][l]['value'] = source['conversations'][l]['value'].replace(x1, x2)

        image = torch.zeros((100 if data_type == 'video' else 1, 768), dtype=torch.float16)

        try:
            feature_path = '{}/{}.npy'.format(self.data_args.feat_folder, source['id'])
            image = np.load(feature_path)
            image = torch.from_numpy(image)
            if data_type == 'image' and len(image.shape) == 1:
                image = image.unsqueeze(0)
        except Exception as e:
            print(e)
            return random.choice(self)

        data_dict = preprocess(
            [source["conversations"]],
            self.tokenizer,
            has_image=True)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        data_dict['image'] = image
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        mask = [torch.ones(size.shape[0]) for size in input_ids]
        pad_mask = torch.nn.utils.rnn.pad_sequence(
            mask,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        pad_mask = pad_mask[:, :self.tokenizer.model_max_length].to(bool)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=pad_mask
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        for key in ("timestamp", "span_label_nn", "timestamp_window"):
            cur = []
            for instance in instances:
                if key in instance:
                    cur.extend(instance[key])
            if cur:
                batch[key] = torch.stack(cur)

        return batch