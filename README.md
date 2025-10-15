<div align="center">
    
<h1>Forest-Chat: Adapting Vision-Language Models for Interactive Forest Change Analysis</h1>

**[James Brock](https://github.com/JamesBrockUoB/)**

<div align="center">
  <img src="resource/conversation_examples.png" width="400"/>
</div>
</div>

## Table of Contents
- [LEVIR-MCI dataset](#LEVIR-MCI-dataset)
- [Forest-Change dataset](#Forest-Change-dataset)
- [Training of MCI model](#Training-of-the-multi-level-change-interpretation-model)
- [Construction of Forest-Chat Agent](#Construction-of-Change-Agent)
- [Citation](#Citation)

## LEVIR-MCI dataset 
- Download the LEVIR_MCI dataset: [LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main).
- This dataset is an extension of our previously established [LEVIR-CC dataset](https://github.com/Chen-Yang-Liu/RSICC). It contains bi-temporal images as well as diverse change detection masks and descriptive sentences. It provides a crucial data foundation for exploring multi-task learning for change detection and change captioning.
    <br>
    <div align="center">
      <img src="resource/dataset.png" width="800"/>
    </div>
    <br>

## Forest-Change dataset
- Data is available in the `Multi_change/data/Forest-Change` folder and can be prepared by running `python preprocess_data.py` in `Multi_change`.
- If you wish to download the original data and create your own captions, then download the images from [here](https://www.kaggle.com/datasets/asheniranga/change-detection-in-forest-covers)
- Name the downloaded folder as `archive`, and place it in the `Multi_change/data` folder
- In the `dataset_utils_notebook.ipynb` file in the project root, run the first three code blocks to format the downloaded data as required
- This should create the `Forest-Change-dataset` folder in the /data directory.
- From here, you can run the captioning app via `streamlit run captioning_app.py` in the `/Multi_change` directory. This will allow you to provide a single human annotated caption, and optionally four rule-based captions per sample. Future work will allow for any number of human captions to be provided.
- Once captioning is complete, the data can be pre-processed as needed by running `python preprocess_data.py` in `/Multi_change`
<br>
  <div align="center">
    <img src="resource/dataset_examples.png" width="800"/>
  </div>
<br>

## Training of the adapted multi-level change interpretation model
The overview of the MCI model as adapted to Forest-Chat:
<br>
    <div align="center">
      <img src="resource/mci_model_forestchat.png" width="800"/>
    </div>
<br>

### Preparation
    
- **Environment Installation**:
    <details open>
    
    **Step 1**: Create a virtual environment named `Multi_change_env` and activate it.
    ```python
    conda create -n Multi_change_env python=3.9
    conda activate Multi_change_env
    ```
    
    **Step 2**: Download or clone the repository.
    ```python
    git clone https://github.com/JamesBrockUoB/ForestChat.git
    cd ./ForestChat/Multi_change
    ```
    
    **Step 3**: Install dependencies.
    ```python
    pip install -r requirements.txt
    ```
    </details>

    **Step 4**: Setup .env file.
    Create a file in the project root folder called `.env` with the following variables:
      - OPEN_AI_KEY: Your OPEN_AI API key - https://platform.openai.com/api-keys
      - SERPER_API_KEY - Your Google Search / Scholar API key - https://serpapi.com/
      - WANDB_USERNAME - Your Weights & Biases username for run logging - https://wandb.ai/site/

- **Download Dataset**:
  <details open>
      
  Link: [LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main). The data structure of LEVIR-MCI is organized as follows:

    ```
    ├─/DATA_PATH_ROOT/Levir-MCI-dataset/
            ├─LevirCCcaptions.json
            ├─images
                 ├─train
                 │  ├─A
                 │  ├─B
                 │  ├─label
                 ├─val
                 │  ├─A
                 │  ├─B
                 │  ├─label
                 ├─test
                 │  ├─A
                 │  ├─B
                 │  ├─label
    ```
    where folder ``A`` contains pre-phase images, folder ``B`` contains post-phase images, and folder ``label`` contains the change detection masks.
    </details>

- **Extract text files for the descriptions of each image pair in LEVIR-MCI**:

    ```
    python preprocess_data.py --dataset LEVIR_MCI-dataset --captions_json LevirCCcaptions.json
    ```

    If you wish to create a subset of LEVIR-MCI that explicitly contains changes to trees called `LEVIR-MCI-Trees`, then add: --keep_only_trees True
    After that, you can find some generated files in `./data/LEVIR_MCI/`. 

- **Download AnyChange**:
  <details open>
  
  Link: [AnyChange](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)

  Place the downloaded model into: `Multi_change/models_ckpt/` with the name unchanged (`sam_vit_h_4b8939.pth`)

  Simplified overview of the AnyChange model:
  <br>
  <div align="center">
        <img src="resource/any_change_simplified_diagram.png" width="800"/>
  </div>

### Train
Make sure you performed the data preparation above. Then, start training as follows:
```python
python train.py --train_goal 2 --savepath ./models_ckpt/
```
This is now configured to use the Forest-Change dataset by default, check commandline arguments and hard-coded constants for parameters that require updating to use LEVIR-MCI. E.g. --data_folder ./data/LEVIR-MCI-dataset/images --list_path ./data/LEVIR_MCI/ --token_folder ./data/LEVIR_MCI/tokens/ --data_name LEVIR_MCI --num_classes 3

### Evaluate
```python
python test.py --checkpoint {checkpoint_PATH}
```
We recommend training the model 5 times to get an average score.

This is now configured to use the Forest-Change dataset by default, check commandline arguments and hard-coded constants for parameters that require updating to use LEVIR-MCI. E.g. --data_folder ./data/LEVIR-MCI-dataset/images --list_path ./data/LEVIR_MCI/ --token_folder ./data/LEVIR_MCI/tokens/ --data_name LEVIR_MCI --num_classes 3

### Inference
Run inference to get started as follows:
```python
python predict.py --imgA_path {imgA_path} --imgB_path {imgA_path} --mask_save_path ./CDmask.png
```
You can modify ``--checkpoint`` of ``Change_Perception.define_args()`` in ``predict.py``. Then you can use your own model, of course, you also can download our pretrained model ``MCI_model.pth`` here: [[Hugging face](https://huggingface.co/lcybuaa/Change-Agent/tree/main)]. After that, put it in `./models_ckpt/`.

This is now configured to use the Forest-Change dataset by default, check commandline arguments and hard-coded constants for parameters that require updating to use LEVIR-MCI. E.g. --data_folder ./data/LEVIR-MCI-dataset/images --list_path ./data/LEVIR_MCI/ --token_folder ./data/LEVIR_MCI/tokens/ --data_name LEVIR_MCI --num_classes 3

## Construction of Change-Agent
<br>
<div align="center">
      <img src="resource/forest-chat-diagram.png" width="800"/>
</div>

- **Agent Installation**:
    ```python
    cd ./ForestChat/lagent-main
    pip install -e .[all]
    ```
- **Run Agent**:

    cd into the ``Multi_change`` folder:
    ```python
    cd ./ForestChat/Multi_change
    ```
    (1) Run Agent Cli Demo:
    ```bash
    # You need to install streamlit first
    # pip install streamlit
    python try_chat.py
    ```
        
    (2) Run Agent Web Demo:
    ```bash
    # You need to install streamlit first
    # pip install streamlit
    streamlit run web_demo.py
    ```
    <br>
    <div align="center">
          <img src="resource/point_query_app_screenshot.png"/>
    </div>

## Acknowledgement
Thanks to the following repositories:

[Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent/tree/main); [AnyChange](https://github.com/Z-Zheng/pytorch-change-models/tree/main/torchange/models/segment_any_change); [lagent](https://github.com/InternLM/lagent)

## License
This repo is distributed under [MIT License](https://github.com/JamesBrockUoB/ForestChat/blob/main/LICENSE). The code can be used for academic purposes only.

