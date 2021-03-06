# MultiModal Fusion Transformer with BERT Encodings for Visual Question Answering

## Abstract
We present MMFT-BERT(MultiModal Fusion Transformer with BERT encodings), to solve Visual Question Answering (VQA) ensuring individual and combined processing of multiple input modalities. Our approach benefits from processing multimodal data (video and text) adopting the BERT encodings individually and using a novel transformer-based fusion method to fuse them together. Our method decomposes the different sources of modalities, into different BERT instances with similar architectures, but variable weights. This achieves SOTA results on the TVQA dataset. Additionally, we provide TVQA-Visual, an isolated diagnostic subset of TVQA, which strictly requires the knowledge of visual (V) modality based on a human annotator's judgment. This set of questions helps us to study the model's behavior and the challenges TVQA poses to prevent the achievement of super human performance. Extensive experiments show the effectiveness and superiority of our method.

Paper link: https://arxiv.org/pdf/2010.14095.pdf

### Bibtex
```
@misc{khan2020mmftbert,
      title={MMFT-BERT: Multimodal Fusion Transformer with BERT Encodings for Visual Question Answering}, 
      author={Aisha Urooj Khan and Amir Mazaheri and Niels da Vitoria Lobo and Mubarak Shah},
      year={2020},
      eprint={2010.14095},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Code and details coming soon...
