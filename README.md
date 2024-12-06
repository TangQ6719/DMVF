# DMVF

This is the implementation of Dual-Modality Visual Feature Flow for Medical Report Generation at MIA-2024.

The repositoriy is based on [M2KT](https://github.com/LX-doctorAI1/M2KT), [CMN](https://github.com/zhjohnchan/R2GenCMN), and [R2Gen](https://github.com/cuhksz-nlp/R2Gen).

## News
Our work [CGFTrans: Cross-Modal Global Feature Fusion Transformer for Medical Report Generation](https://github.com/TangQ6719/CGFTrans) at JBHI-2024 [paper](https://doi.org/10.1109/JBHI.2024.3414413)).

## Requirements

- `torch==1.7.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`

## Datasets
- Download IU X-ray, MIMIC-CXR, LGK, and COV-CTR datasets, and place in `data` directory within the project folder.
  - [Download IU X-ray Dataset](https://iuhealth.org/find-medical-services/x-rays)
  - [Download MIMIC-CXR- Dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
  - [Download LGK Dataset](https://pan.quark.cn/s/e9cf4c649b8f) Extraction code：LNTL
  - [Download COV-CTR Dataset](https://github.com/mlii0117/COV-CTR)

## Contact
* If you have any questions, please post it on github issues or email at tangquan6719@gmail.com

## Reference
* [https://github.com/LX-doctorAI1/M2KT](https://github.com/LX-doctorAI1/M2KT)
* [https://github.com/zhjohnchan/R2GenCMN](https://github.com/zhjohnchan/R2GenCMN)
* [https://github.com/cuhksz-nlp/R2Gen](https://github.com/cuhksz-nlp/R2Gen)
