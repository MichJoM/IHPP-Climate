We are releasing Italian HyperPartisan and Pragmatics in CLIMATE News (IHPP-Climate), a dataset for hyperpartisan italian news on climate change.
The dataset is divided into train (IHPPC_train.json) and test (IHPP_test.json), that can be found in /data.

The dataset can be used to reproduce our findings. 


##### Models tested:

| Model Name                             | Model Architecture | Hugging Face ID                                 | Note                    |
|----------------------------------------|--------------------|--------------------------------------------------|-------------------------|
| GPT-4o mini                            | Decoder            | -                                               | API                     |
| Llama3.1-70B-Instruct                  | Decoder            | meta-llama/Meta-Llama-3.1-70B-Instruct          | Access request required |
| bert-base-italian-uncased              | Encoder            | dbmdz/bert-base-italian-uncased                 | -                       |
| bert-base-italian-xxl-uncased          | Encoder            | dbmdz/bert-base-italian-xxl-uncased             | -                       |
| sentence-bert-base-italian-uncased     | Encoder            | nickprock/sentence-bert-base-italian-xxl-uncased| -                       |
| sentence-bert-base-italian-xxl-uncased | Encoder            | nickprock/sentence-bert-base-italian-xxl-uncased| -                       |
| bert-base-multilingual-uncased         | Encoder            | google-bert/bert-base-multilingual-uncased      | -                       |


Integrated Gradient:
To run the experiment on the integrated gradient:
1. Download the models described in the table.
2. Run `attn_comparative`.py

Fine-Tuning:
1. Run the `ft.py`.

Fine-Tuning with Embeddings`:
1. Run the `FT_all.py`

Ablation study: 
1. Run `ablation.py`.

Few shot (ICL) experiment:
1. Generate DPP points executing the file DPP.ipynb or select ./data/IHPP_DPP_LT.json.
2. Run `Llama_classification.py`
3. Run `GPT_classification.py`
