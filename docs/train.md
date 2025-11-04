# Training VideoExpert
VideoExpert is a general-purpose MLLM, designed to be well-suited for a variety of temporal-sensitive video tasks.

Before starting training, you need to prepare the following:

* The dataset and the corresponding video clips. Please refer to [data.md](data.md) for more details.

* Download [clip](https://cloud.tsinghua.edu.cn/d/6db5d02883124826aa6f/?p=%2Fcheckpoints&mode=list) and [Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) weights, and place them into the 'checkpoints' directory.

```markdown
- VideoExpert
    - Checkpoints
        - clip
        	- ViT-L-14.pt
        - vicuna-7b-v1.5
        	- pytorch_model-00001-of-00002.bin
        	- ...
    - Datasets
        - Eval_Data
            - ...
        - Pretrain
            - ActivityNet.json
            - InterVid.json
            - Next_QA_Train.json
        - PreTrain_Data
            - ActivityNet
            - InterVideo
            - Next_Videos
            
    - VideoExpert_Script
    	- Hybrid_Pretraining.sh
    	- ...

```

Please make sure to modify  '--feat_folder' in the script to the corresponding feature folder for each stage.

```shell
cd VideoExpert
bash VideoExpert_Script/Hybrid_Pretraining.sh
```


Lastly, we provide the pre-trained VideoExpert model, which can be downloaded from [here](https://pan.baidu.com/s/1fWexF8C-lq1UblKUBvKm2A) (Code:ga9n). 