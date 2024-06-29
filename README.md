# MultiModal Fusion Transformer with BERT Encodings for Visual Question Answering (EMNLP 2020 Findings)

## Abstract
We present MMFT-BERT(MultiModal Fusion Transformer with BERT encodings), to solve Visual Question Answering (VQA) ensuring individual and combined processing of multiple input modalities. Our approach benefits from processing multimodal data (video and text) adopting the BERT encodings individually and using a novel transformer-based fusion method to fuse them together. Our method decomposes the different sources of modalities, into different BERT instances with similar architectures, but variable weights. This achieves SOTA results on the TVQA dataset. Additionally, we provide TVQA-Visual, an isolated diagnostic subset of TVQA, which strictly requires the knowledge of visual (V) modality based on a human annotator's judgment. This set of questions helps us to study the model's behavior and the challenges TVQA poses to prevent the achievement of super human performance. Extensive experiments show the effectiveness and superiority of our method.

[Paper](https://aclanthology.org/2020.findings-emnlp.417.pdf)

This code repo is modified from [TVQA](https://github.com/jayleicn/TVQA) codebase. 

## dataset
Follow instructions on TVQA github repo to download data. 
Alternatively, the processed data files are provided [here](https://drive.google.com/file/d/1RO_BJ8Lz3NST4RNcCF-e5oNrvrdrDt1q/view?usp=sharing). download the files, and extract them. 

`
unzip data.zip
`

Copy the extracted files into the data directory.

Download tvqa vocab and indexing dictionaries from [here](https://drive.google.com/file/d/1rkWn0wer_fRksg8OrDXNoMwK4GfKD52Z/view?usp=sharing) and unzip them in the cache folder by using `unzip cache.zip` inside `cache/` directory.

### Training files
`main_dict_multiple_losses.py` is the main file for training.
`tvqa_vqa_2bert_bertfusion_sub.py` is the model definition. 
`tvqa_dataset_vqa_bert_attn.py` has the dataloader for TVQA dataset. 

<!-- To train the system, run the following command: 
`python main_dict_multiple_losses.py --input_streams vcpt sub` -->




### Bibtex
```
@inproceedings{urooj-etal-2020-mmft,
    title = "MMFT-BERT: Multimodal Fusion Transformer with BERT Encodings for Visual Question Answering",
    author = "Urooj, Aisha  and
      Mazaheri, Amir  and
      Da vitoria lobo, Niels  and
      Shah, Mubarak",
    editor = "Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.417",
    doi = "10.18653/v1/2020.findings-emnlp.417",
    pages = "4648--4660",
}

```


