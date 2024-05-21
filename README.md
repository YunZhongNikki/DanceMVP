## DanceMVP: Self-Supervised Learning for Multi-Task Primitive-Based Dance Performance Assessment via Transformer Text Prompting

✨This is the official implementation of AAAI 2024 paper: [DanceMVP](https://ojs.aaai.org/index.php/AAAI/article/view/28893)

### Introduction

In this paper, we formulated a novel research problem of automatic human dance performance evaluation to help people improve their sensorimotor skills. Specifically, we propose DanceMVP: multi-task dance performance assessment via text prompting that solves three related tasks - (i) dance vocabulary recognition, (ii) dance performance scoring and (iii) dance rhythm evaluation.

![p](https://github.com/YunZhongNikki/ImperialDance-Dataset/blob/main/new_frameworkk_large-1.png)

By releasing this code, we aim to facilitate further research and development in human motion analysis, motion generation, action quality assessment and skill assessment. If you find this work useful in your research, please feel free to cite our paper as a reference.

### Preparation

#### 1. Prepare the Environment Requirements
To run the code, please install the dependency libraries by using the following command:
```
pip3 install -r requirements_20230816.txt
```

#### 2. Prepare the Pre-trained Model
The dance motion-music model is pretrained by using our [ImperialDance Dataset](https://github.com/YunZhongNikki/ImperialDance-Dataset/tree/main).

You can download the pre-trained model here [model](https://drive.google.com/drive/folders/13D6fEwO97Uhs8kRckjajtZFVXNBtdbl9?usp=drive_link), and place it under the folder './ckpts/'.


#### 3. Prepare the Dataset
If you want to train the model by yourself, place the dataset under the folder './datasets/dance_level/'. The dataset can be found at [ImperialDance Dataset](https://github.com/YunZhongNikki/ImperialDance-Dataset/tree/main).

### Pre-train
You can utilize the following code to run the pre-train process. The log files and model checkpoint will be stored in './runs/'.
```
python3 level_contrastive_graph.py
```

### Downstream Task
The downstream evaluation task can be run by the following command.
```
python3 test_our_text_prompt.py
```

### Citation
```
@inproceedings{zhong2024dancemvp,
  title={DanceMVP: Self-Supervised Learning for Multi-Task Primitive-Based Dance Performance Assessment via Transformer Text Prompting},
  author={Zhong, Yun and Demiris, Yiannis},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={9},
  pages={10270--10278},
  year={2024}
}
```

