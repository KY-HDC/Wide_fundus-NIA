# Classification using Wide Fundus images

건양대학병원에서 제공하는 광각안저영상(Wide Fundus Image)를 사용하여 황반변성, 당뇨망막병증, 망막정맥폐쇄와 정상을 구분하는 학습 모델을 구현하였다. 두 명의 안과 의사의 교차검증을 통한 데이터셋에 대해서 학습을 진행하였다. Google에서 2016년도에 발표한 [Xception](https://arxiv.org/pdf/1610.02357.pdf)을 customized한 모델을 구현해 사용하였다.

## Data Preprocessing and Augmentation
광각안저영상에 대해 영상 1장을 96장으로 증폭하여 사용함
질환별로 다른 필터를 적용하여 영상에 특화된 전처리를 시행
Preprocessing_AMD & Preprocessing_DMR&RVO 에서 실제 구현 세부 내용을 확일 할 수 있음

### Augmentation
&nbsp; | 증강 종류 | 설명 
:----: | :-----: | :-:
1 | 영상 필터 종류 | Bilateral, Gaussian, Histogram Equalization, Median, Sharpening (황반변성 - Sobel, Scharr 추가)
2 | 영상 확대 여부 | 125% 확대 후 center를 중심으로 299*299 crop (황반변성 - 60, 120, 180, 240, 300도)
3 | 영상 회전 여부 | 90, 180, 270도 회전


## Performance Estimation
광각안저영상에 해당하는 질환별 진단모델의 성능은 기존 주어진 데이터 세트에 대해서 10 fold crosss validation을 진행하였으며, 각 Fold에 대한 성능 지표로는 정확도(Accuracy), 민감도(Sensitivity), 특이도(Specificity), AUROC를 사용하여 학습하고 평가에 사용하였다.
 
## Prerequisites
- **Python** ≥ `3.6`
- **Keras**     ≥            `2.2.4`
- **numpy**           ≥      `1.16.4`
- **opencv**    ≥            `3.2.0`
- **pandas**        ≥        `0.24.2`
- **sklearn**      ≥         `0.21.1`
