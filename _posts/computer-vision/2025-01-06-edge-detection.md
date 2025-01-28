---
title: Edge detection
tags: [CV, C++]
category: Computer-Vision
toc: true 
math: true
img_path: /assets/posts/edge-detection/
---

`엣지(edge) 검출`은 객체의 경계를 찾는 방법으로 객체 판별 전처리 과정으로 사용한다. 본 글은 대표적인 엣지(이하 경계) 검출에 필요한 수학적 배경과 알고리즘에 대해 설명한다.

---

## 미분과 변화량

경계 검출의 핵심은 **변화**를 찾는 것이다. 객체와 배경은 밝기 차이가 있을 것이라고 가정한다. 밝기 변화가 일정 수준을 넘어가면 경계로 예측한다. 이미지가 복잡하면 잘못 검출될 가능성도 있지만 합리적인 아이디어라고 볼 수 있다.

![객체와 배경 밝기 비교](color-diff.png)

그렇다면 **변화**를 정의해야 한다. 수학에서 변화율은 **미분**으로 정의한다. 연속 함수 $f(x)$에 대해 미분은 아래와 같다.

$$f'(x) = \cfrac{df}{dx}=\lim_{\bigtriangleup x \to 0}\cfrac{f(x+\bigtriangleup x)-f(x)}{\bigtriangleup x}$$

$\bigtriangleup x$는 **변화량**이다. **미분값**은 변화량이 0에 가까워질 때 함수 값의 차이를 뜻한다. 쉽게 말해, 특정 시점에서 함수 값의 변화로 볼 수 있다. 위 파란 그래프는 함수 $f(x)$, 아래 빨간 그래프는 $f(x)$를 미분한 $f'(x)$다. 변화가 멈춘 순간에 미분값은 0이 된다. 급격한 변화가 발생하면 미분값이 0에서 멀어진다.

![미분 함수 그래프](derivative-graph.png)

### 이산 함수 미분

위에서 살펴본 미분법은 함수가 연속적일 때 적용가능하다. 이미지는 독립된 픽셀로 이루어져 있다. 따라서 **이산 값에 대한 미분을 다시 정의**한다.

$$f'(x) = \cfrac{df}{dx}\approx \cfrac{f(x+\bigtriangleup h)-f(x)}{\bigtriangleup h}$$

여기서 변화량 $\bigtriangleup h$는 픽셀 간의 거리를 뜻한다.

그리고 이미지는 2차원 좌표 $(x,y)$를 가진다. 따라서, x 방향과 y 방향에 대한 미분을 모두 정의해야 한다.

$$f'_x(x,y) = \cfrac{df}{dx}\approx \cfrac{f(x+\bigtriangleup h,y)-f(x,y)}{\bigtriangleup h}$$

$$f'_y(x,y) = \cfrac{df}{dy}\approx \cfrac{f(x,y+\bigtriangleup h)-f(x,y)}{\bigtriangleup h}$$

이를 시각화해보면 이해가 쉽다. 인접한 픽셀과의 차를 구하는 식이다.

![이산 미분 마스크](discrete-derivative-mask.png)

$$f'_x\approx \cfrac{f(x+1,y)-f(x,y)}{1}=59 - 30$$

$$f'_y\approx \cfrac{f(x,y+1)-f(x,y)}{1}=87 - 30$$

### 중앙 차분

중앙 차분은 인접한 두 픽셀의 미분 값을 구하는 방식이다.

![중앙 차분 마스크](centered-diff-mask.png)

$$f'_x\approx \cfrac{f(x+1,y)-f(x-1,y)}{2}$$

$$f'_y\approx \cfrac{f(x,y+1)-f(x,y-1)}{2}$$

정의대로라면 픽셀 간 거리인 $h$가 2이므로, 2로 나누어야 한다. 하지만 우리가 필요한 건 상대적인 크기다. 물체와 배경의 밝기가 상대적으로 얼마나 다른가이다. 따라서 2로 나누는 과정을 생략하고 약식으로 계산한다.

$$f'_x\approx f(x+1,y)-f(x-1,y)=59-17$$

$$f'_y\approx f(x,y+1)-f(x,y-1)=87-40$$

거창한 내용 같지만 결국은 **인접한 두 픽셀의 차**를 구하는 식이 된다.

### 행렬 연산

행렬 연산을 이용하면 효율적으로 연산할 수 있다. x 방향 미분 식을 다시 살펴보자.

$$f'_x\approx f(x+1,y)\cdot 1 + f(x,y)\cdot 0 - f(x-1,y)\cdot 1$$

$$f'_x\approx\begin{bmatrix} f(x-1,y) & f(x,y) & f(x+1,y) \end{bmatrix}\begin{bmatrix}-1 \\ 0\\ 1 \end{bmatrix}$$

y 방향도 같은 방법으로 행렬을 만들 수 있다.

![마스크 연산 예시](mask-computation.png)

정리하면, $f(x,y)$와 인접한 픽셀의 변화량을 통해 현재 위치가 경계인지 판별할 수 있다. 이때 효율적인 연산을 위해 행렬을 이용한다.

---

## Gradient 정의

미분은 gradient를 설명하기 위한 빌드업이었다. **Gradient**란 x 방향과 y 방향의 미분값을 나타내는 벡터이다.

$$\bigtriangledown f=\begin{bmatrix} f_x \\ f_y \end{bmatrix}=f_x i + f_y j$$

$i,j$는 각 방향에 대한 단위 벡터를 뜻한다. 벡터의 크기는 $\parallel \bigtriangledown f\parallel $, 벡터의 방향은 $\theta$로 표현한다.

$$\parallel \bigtriangledown f\parallel =\sqrt{f_x^2+f_y^2}$$

$$\theta =tan^{-1}(\cfrac{f_y}{f_x})$$

![gradient 벡터 예시](grad-vectors.png)

이미지 일부를 확대한 뒤 2차원 공간에 gradient 벡터를 나타냈다. 경계로 판단되는 부분은 벡터의 크기가 매우 크다. 벡터의 방향은 변화가 발생하는 방향을 나타낸다. 다시 말해, **벡터에 수직인 방향이 경계**라고 볼 수 있다. 확실히 경계가 아니라고 판단되는 곳은 크기와 방향 모두 0을 가진다.

---

## 다양한 마스크

앞서 행렬 연산을 이용한다고 했다. 이 행렬을 **마스크(mask)**, 필터(filter) 또는 커널(kernel) 등으로 부른다. 본 글에서는 "마스크"로 통일하겠다. 앞서 \[-1 0 +1\] 형태의 단순한 마스크를 소개했다. 그 외에 더 정교한 경계 검출을 위해 여러 마스크가 개발되었다.

### Sobel

**Sobel 마스크**는 가장 대표적인 마스크다. 인접한 두 픽셀뿐만 아니라 근접한 픽셀까지 고려한다.

![sobel 마스크](sobel-mask.png)

앞서 벡터의 크기를 통해 경계가 맞는지 확인한다고 했다. 하지만 의미없는 노이즈도 섞여 있을 수 있다. 따라서 벡터가 특정 범위를 넘어서면 경계로 판별한다. 이때 기준이 되는 값을 **threshold** 또는 **임계값**이라고 한다. threshold는 상황에 맞게 직접 설정해주어야 한다.

![sobel 결과 이미지](sobel-result.png)

```cpp
Mat dx, dy;
Sobel(img, dx, CV_32FC1, 1, 0);
Sobel(img, dy, CV_32FC1, 0, 1);

Mat mag_float, mag;
magnitude(dx, dy, mag_float);
mag_float.convertTo(mag, CV_8UC1);

int threshold = 150;
Mat edge = mag > threshold;

imshow("edge", edge);
```

### Scharr

**Scharr 마스크**는 인접한 픽셀에 더 큰 가중치를 준다. 따라서 Sobel보다 변화에 더 민감하다.

![scharr 마스크](scharr-mask.png)

theshold를 높게 설정했음에도 신발 얼룩까지 포함하는 모습을 보인다. 얼룩도 밝기 변화가 있는 영역이기 때문이다.

![scharr 결과 이미지](scharr-result.png)

```cpp
Scharr(img, dx, CV_32FC1, 1, 0);
Scharr(img, dy, CV_32FC1, 0, 1);

magnitude(dx, dy, mag_float);
mag_float.convertTo(mag, CV_8UC1);

int threshold = 250;
Mat edge = mag > threshold;

imshow("edge", edge);
```

---

## Canny edge detector

**Canny 검출기**는 단순한 마스크보다 더 정확한(tight) 테두리를 검출하기 위해 개발되었다.

1. Gaussian Filter
2. Gradient
3. NMS: non-maximum  suppression
4. Double thresholding
5. Hysteresis edge tracking

### Gaussian Filter

**Gaussian Filter는 가우시안 정규분포를 활용해 노이즈를 제거하는 과정**이다. 노이즈는 주변과 다른 형태를 띠는 값이기 때문에 미분을 수행했을 때 큰 값으로 나타날 수 있다. 따라서 노이즈의 영향을 줄이기 위해 필터를 사용한다. 평균이 0, 표준편차가 $\sigma$라고 할 때, 2차원 가우시안 분포는 아래와 같다.

$$G_{\sigma_x\sigma_y}(x,y)=\cfrac{1}{2\pi\sigma_x\sigma_y}e^{-(\cfrac{x^2}{2\sigma^2_x}+\cfrac{y^2}{2\sigma^2_y})}$$

![가우시안 분포 시각화](gaussian-graph.png)

가우시안 필터를 사용하면 중앙에 비교적 큰 값이 곱해지고, 주변은 작은 값이 곱해진다. 주변 상황을 약하게 반영하는 과정에서 비교적 완만한 값이 만들어진다. 따라서 부드러운 이미지를 만드는 블러 효과로 사용한다.

평균이 0이고 표준편차가 $\sigma$일 때, $[-4\sigma ,4\sigma]$ 사이에 99.99%의 값이 들어가 있기 때문에 마스크 크기는 $8\sigma +1$이나 그보다 작은 크기를 사용한다.

![가우시안 필터 결과 이미지](gaussian-result.png)

동일한 조건에서 5 x 5 가우시안 필터를 적용했을 때와 적용하지 않았을 때 검출된 경계의 모습이다. 신발 발등의 불규칙한 얼룩이 제거되었다.

### Gradient

Sobel 마스크를 활용해 gradient를 계산한다. 하지만 앞서 소개한 L2 norm을 이용한 크기 계산은 과정이 복잡하다. 따라서 간단한 **L1 norm**을 사용해 단순하게 연산하다.

$$\parallel \bigtriangledown f\parallel \approx |f_x|+|f_y|$$

추가로 gradient 방향도 함께 계산한다. 계산된 방향은 4가지 방향\[0, 45, 90, 135\]으로 단순화할 수 있다. 각 픽셀이 사각형의 형태로 붙어 있기 때문이다.

![gradient 방향 시각화](grad-direction.png)

### NMS: Non-maximum suppression

Sobel을 거친 gradient는 비슷한 지역에서 여러 경계를 만들기도 한다. 이 현상 때문에 일부 경계가 두껍게 나타난다.

![두꺼운 경계의 픽셀 확대](bold-edge.png)

**NMS: non-maximum suppression**은 경계로 판단되는 픽셀 중 가장 확실한 픽셀만 선택한다. gradient 방향으로 인접한 두 픽셀을 비교한다. 그리고 가운데 픽셀이 가장 클 경우 경계로 사용하고, 그렇지 않을 경우 0으로 처리한다.

![nms 실행 과정](nms.png)

이 과정을 통해 겹쳐있는 경계 영역 중 정확한 경계를 가려낸다.

![nms 결과 이미지](nms-result.png)

동일한 조건에서 NMS를 실행했을 때와 실행하지 않았을 때의 모습이다. 겹쳐있던 선이 제거되었다.

### Double thresholding

**Double thresholding**은 임계값 2개를 이용해 경계를 판별한다. 높은 임계값을 $T_{high}$, 낮은 임계값을 $T_{low}$라고 하자.

- $\parallel \bigtriangledown f\parallel  \ge T_{high}$: 확실한 경계로 판별
- $ \parallel \bigtriangledown f\parallel  \le T_{low}$: 경계가 아님
- $else$: edge tracking 진행

두 임계값 사이에 있는 픽셀은 추가 검사를 진행한다.

### Hysteresis edge tracking

**Hysteresis edge tracking**은 확실한 경계를 넓혀가는 방식으로 경계를 추가한다.

![edge tracking 실행 과정](edge-tracking.png)

확실하게 경계로 판별된 픽셀에 대해 주변 픽셀을 검사한다. 만약 주변 픽셀 중 $T_{high}$보다는 작지만, $T_{low}$보다 큰 값이 있다면 경계로 판별한다. 다시 말해, $T_{high}$와 $T_{low}$ 사이 값 중 $T_{high}$와 연결된 픽셀은 경계로 인정한다. 반면에 $T_{low}$와 연결된 사이 값은 경계로 인정하지 않는다. tracking을 통해 연결된 테두리를 추가로 찾을 수 있다.

![edge tracking 결과](edge-tracking-result.png)

### 정리

Canny 알고리즘의 각 단계가 어떤 과정으로 진행되고, 적용했을 때와 적용하지 않을 때의 결과 차이를 알아보았다. 전체 과정을 정리하면 아래와 같다.

![gaussian-gradient-nms-edge tracking](canny-result.png)

각 단계를 거친 이미지 행렬이다. OpenCV는 `Canny` 함수를 통해 이 복잡한 과정을 한 번에 처리할 수 있다.

```cpp
Canny(img, dst, 100, 200);
```

만약 구현 과정이 궁금하다면 [Github(denev6/deep-learning-codes)](https://github.com/denev6/deep-learning-codes/blob/main/OpencvCpp/src/edge.cpp#L112)를 참고하면 된다.
