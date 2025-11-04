# L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning

Implementation for the paper:  
[**L3A: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning**](https://proceedings.mlr.press/v267/zhang25y.html)  
**Authors:** Zhang, Xiang and He, Run and Jiao, Chen and Fang, Di and Li, Ming and Zeng, Ziqian and Chen, Cen and Zhuang, Huiping  
**Conference:** Proceedings of the 42nd International Conference on Machine Learning, 2025


---

## Abstract
Class-incremental learning (CIL) enables models to learn new classes continually without forgetting previously acquired knowledge. Multi-label CIL (MLCIL) extends CIL to a real-world scenario where each sample may belong to multiple classes, introducing several challenges: label absence, which leads to incomplete historical information due to missing labels, and class imbalance, which results in the model bias toward majority classes. To address these challenges, we propose Label-Augmented Analytic Adaptation (L3A), an exemplar-free approach without storing past samples. L3A integrates two key modules. The pseudo-label (PL) module implements label augmentation by generating pseudo-labels for current phase samples, addressing the label absence problem. The weighted analytic classifier (WAC) derives a closed-form solution for neural networks. It introduces sample-specific weights to adaptively balance the class contribution and mitigate class imbalance. Experiments on MS-COCO and PASCAL VOC datasets demonstrate that L3A outperforms existing methods in MLCIL tasks. Our code is available at https://github.com/scut-zx/L3A.

---

## Environment Setup

Create a Conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate l3a
   ```

---

## Dataset Preparation

1. **Download Datasets:**
   - **MS-COCO 2014:** Download and place the dataset under `./datasets/coco`.
   - **PASCAL VOC 2007:** Download and place the dataset under `./datasets/VOCdevkit`.

2. **Modify Configuration Files:**
   Update the dataset paths in the corresponding `.yaml` files under the `config` directory.

3. **Download Pretrained Model:**
   Download the **TResNetM** model pretrained on ImageNet 21k from [TResNetM Pretrained Model](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ASL/MS_COCO_TRresNet_M_224_81.8.pth).  
   Place the model in the `./pretrained_models` directory and rename it to `tresnet_m_224_21k.pth`.

---

## Training

All commands should be executed from the project root directory.

### Train on MS-COCO
```bash
bash train_coco.sh
```

### Train on PASCAL VOC
```bash
bash train_voc.sh
```

---

## Results

Training results will be saved in the `logs/` directory.  
- Detailed logs can be found in `logs/**/log/log.txt`.  
- Models for each incremental stage are stored under the `saved_models/` directory.

---

## Acknowledge

We thank the authors of [KRT](https://github.com/Songlin-Dong/KRT-MLCIL/tree/main) for their contributions to multi-label class incremental learning, which inspired and supported the development of our L3A framework.

---

## Citation

If you find our work useful for your research, please cite our paper:
```bibtex
@InProceedings{L3A_Zhang_ICML2025,
  title     = 	 {{L}3{A}: Label-Augmented Analytic Adaptation for Multi-Label Class Incremental Learning},
  author    =   {Zhang, Xiang and He, Run and Jiao, Chen and Fang, Di and Li, Ming and Zeng, Ziqian and Chen, Cen and Zhuang, Huiping},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages     = 	 {74938--74949},
  year      = 	 {2025},
  editor    = 	 {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume    = 	 {267},
  series    = 	 {Proceedings of Machine Learning Research},
  month     = 	 {13--19 Jul},
  publisher =   {PMLR}
}

