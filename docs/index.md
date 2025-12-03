---
layout: paper
paper: "Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation"
github_url: https://github.com/grip-unina/WaveRep-SyntheticVideoDetection
authors:  
  - name: Riccardo Corvi
    link: https://www.grip.unina.it/members/corvi
    index: 1
  - name: Davide Cozzolino
    link: https://www.grip.unina.it/members/cozzolino
    index: 1
  - name: Ekta Prashnani
    link: https://prashnani.github.io/
    index: 2
  - name: Shalini De Mello
    link: https://research.nvidia.com/person/shalini-de-mello
    index: 2
  - name: Koki Nagano
    link: https://luminohope.org/
    index: 2
  - name: Luisa Verdoliva
    link: https://www.grip.unina.it/members/verdoliva
    index: 1
affiliations: 
  - name: University Federico II of Naples, Italy
    index: 1
  - name: NVIDIA
    index: 2
links:
    arxiv: https://arxiv.org/abs/2506.16802
    paper: https://openreview.net/forum?id=dOGXKBL7IE
    code: https://github.com/grip-unina/WaveRep-SyntheticVideoDetection
---

<center>
 <img src="./teaser.svg" alt="teaser" width="100%" style="transform: scale(1.15);" />
</center>

In this work, first, we study different generative architectures, searching and identifying discriminative features that are unbiased, robust to impairments, and shared across models. Then, we introduce a novel forensic-oriented data augmentation strategy based on the wavelet decomposition and replace specific frequency-related bands to drive the model to exploit more relevant forensic cues. Our novel training paradigm improves the generalizability of AI-generated video detectors, without the need for complex algorithms and large datasets that include multiple synthetic generators. To evaluate our approach, we train the detector using data from a single generative model and test it against videos produced by a wide range of other models. Despite its simplicity, our method achieves a significant accuracy improvement over state-of-the-art detectors and obtains excellent results even on very recent generative models, such as NOVA and FLUX.

## News
* 2025-11-28: demo release.
* Coming Soon: Test set

## Bibtex

```
@InProceedings{corvi2025seeing,
    title = {Seeing What Matters: Generalizable AI-generated Video Detection with Forensic-Oriented Augmentation},
    author = {Riccardo Corvi and Davide Cozzolino and Ekta Prashnani and Shalini De Mello and Koki Nagano and Luisa Verdoliva},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year = {2025},
    url = {https://openreview.net/forum?id=dOGXKBL7IE}
}
```

## Acknowledgments

This work has received funding from the European Union under the Horizon Europe vera.ai project,
Grant Agreement number 101070093, and was partially supported by SERICS (PE00000014) under
the MUR National Recovery and Resilience Plan, funded by the European Union - NextGenerationEU.
We thank David Luebke for early discussions on the project.
