We provide the pre-training dataset and the corresponding annotations, which can be downloaded from [here](https://pan.baidu.com/s/1fWexF8C-lq1UblKUBvKm2A) (Code:ga9n). 

The pre-training dataset consists of a subset from the following datasets:

* Internvid: https://huggingface.co/datasets/OpenGVLab/InternVid
* ActivityNet: http://activity-net.org/download.html
* NextQA: https://github.com/doc-doc/NExT-QA

Please download the corresponding video clips included in the pre-training dataset independently.


---
---



The data from Internvid and ActivityNet is organized in the following JSON format:

* source: The original sources of the dataset.
* id: The ID of the video in the original dataset.
* conversations: The conversation data used for training. We utilize special tokens  (such as `<s0>`, `<e0>`) to indicate timestamps. These need to be replaced during training using information from the 'meta' field.
* meta:
  * split: refers to the extraction of a segment from the original  video as input, with two numbers denoting the start and end timestamps  of the extraction. If this field does not exist, the entire video is  used as input. 
  * duration: indicates the length of the input video.
  * token: represents the timestamp of special tokens appearing in  'conversations'.

Here is an example:

```json
{
    "source": "internvid",
    "id": "3n3oCNerzV0",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nI'd appreciate it if you could provide a detailed account of the events that occurred at different timestamps in the video."
        },
        {
            "from": "gpt",
            "value": "From <s0> to <e0>, woman is counting money with a pen on a white table. From <s1> to <e1>, two people shaking hands in front of a desk."
        }
    ],
    "meta": {
        "split": [
            129.8,
            184.0
        ],
        "duration": 54.2,
        "token": {
            "<s0>": 0,
            "<e0>": 9.4,
            "<s1>": 49.4,
            "<e1>": 53.2
        }
    }
}

```