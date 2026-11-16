
## ⚙️ Installation

```bash
git clone https://github.com/jasongief/TGS-Agent.git
cd TGS-Agent
```

### For Think Phase
```
conda env create -f think_environment.yml
conda activate think
```
*Alternative: you may also refer to [Crab](https://github.com/GeWu-Lab/Crab) for environment installation*

### For Ground-Segment Phase
```
cd ground_segment_scripts

git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2

conda env create -f dinosam2_environment.yml
conda activate dino
```
*Alternative: you may refer to [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2) for environment installation* 


```
cd ../TGS-Agent
```

## 🤗 Setup

### Datasets

- Download the official Ref-AVSBench dataset from [here](https://github.com/GeWu-Lab/Ref-AVS) and put them in ```./REFAVS```. The metadata (csv file) should also be copyed to ```./R2AVSBench```
- Download our instruction tuning data for Ref-Thinker training from [here](https://huggingface.co/datasets/Jinxing1/TGSAgent-FT-data/tree/main) and put the json file into ```./R2AVSBench```.
- Download metadata of our R2-AVSBench from [here](https://drive.google.com/drive/folders/1Qz7MxBs7IpxgcTH8CaUsU3i9d366gRhM) and put the csv file into ```./R2AVSBench```.


### Pretrained Backbones
Download the necessary pre-trained backbones and put them in ```./pretrained_weights```, including

**Multimodal Encoder Weights:**
- download visual encoder [openai-clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- download audio encoder [BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2](https://github.com/microsoft/unilm/blob/master/beats/README.md)

**LLM Weights:**

download [LLaMA-2-Chat-HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

**Pretrained Multimodal Projector**

- download pretrained audio projector: [audio pretrain checkpoint](https://huggingface.co/ahsgdxhs/Crab/blob/main/audio_pretrain.bin)
- download pretrained visual projector: [visual pretrain checkpoint](https://huggingface.co/ahsgdxhs/Crab/blob/main/visual_pretrain.bin), 



### Checkpoints

Download the following checkpoints:
- download our pretrained **[Ref-thinker](https://huggingface.co/Jinxing1/TGS-Agent/tree/main)** and put it into ```results_real```.
- run the following scripts to prepare **GroundingDINO** weights:
``` 
cd ./ground_segment_scripts/Grounded-SAM-2/gdino_checkpoints
bash download_ckpts.sh
```
- run the following scripts to prepare **SAM2** weights:
``` 
cd ./ground_segment_scripts/Grounded-SAM-2/checkpoints
bash download_ckpts.sh
```


## 📌 Getting Started

### Train Ref-Thinker
```
cd TGS-Agent
conda activate think
bash scripts/finetune/finetune_hyperlora.sh
```
### Test Ref-Thinker
```
cd TGS-Agent
conda activate think
bash scripts/finetune/inference_hyper_lora.sh
```
This generates the object-aware reasoning chain for each given reference from default Ref-AVSBench. You may change the test meta csv path for evaluating our proposed R^2-AVSBench.
After obtaining the fine-grained and simplified object description, we can start the subsequent Ground and Segment phase.

### Ground-Segment
```
cd ground_segment_scripts
conda activate dino
```

-  inference on Ref-AVSBench prompted by Ref-Thinker
```
python ground_segment_with_object_text_after_thinking_for_RefAVSBench.py
```

-  inference on Ref-AVSBench prompted by Original raw reference
```
python ground_segment_with_direct_reference_of_RefAVSBench.py
```

-  inference on R^2-AVSBench prompted by Ref-Thinker
```
python ground_segment_with_object_text_after_thinking_for_R2AVSBench.py
```

-  inference on R^2-AVSBench prompted by Original raw reference
```
python ground_segment_with_direct_reference_of_R2AVSBench.py
```


## Acknowledgement

We thank the [Crab](https://github.com/GeWu-Lab/Crab) and  [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2) for their open-source, which help a lot in this project.



