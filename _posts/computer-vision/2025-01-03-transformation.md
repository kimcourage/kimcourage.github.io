---
title: 이미지 변환 행렬과 OpenCV
tags: [CV, C++]
category: Computer Vision
toc: true 
math: true
img_path: /assets/posts/transform
---

이미지 행렬의 이동, 확대, 축소 등 **기하학적 변환**에 대해 다룬다. C++로 작성한 OpenCV 코드를 사용한다. 원본 이미지 좌표는 $(x, y)$로, 변환된 이미지 좌표는 $(x',y')$로 표현한다. 간결한 코드를 위해 네임스페이스를 생략하며, 이미지를 읽는 과정도 생략한다. 코드에서 img는 원본 이미지, dst는 변환된 이미지이다.

원본 이미지의 모습이다.

![img](original.jpg)

OpenCV는 `warpAffine`과 `perspectiveTransform` 메서드를 지원한다.

- **warpAffine**: 어파인 변환 행렬을 이용
- **perspectiveTransform**: 투시 변환 행렬을 이용

---

## 이동 변환

![resized img](translation.png)

`이동(translation)` 변환은 이미지 좌표를 x, y 방향으로 이동(shift)한다. 평행 이동은 간단한 **덧셈**으로 구현 가능하다.

$$x' = x + \bigtriangleup x$$

$$y' = y + \bigtriangleup y$$

반복문을 돌며 값을 하나씩 더하면 연산 비용이 매우 크다. 따라서 행렬 연산으로 처리한다.

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & 0 & \bigtriangleup x \\ 0 & 1 & \bigtriangleup y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

변화 값을 더해주기 위해 \[x, y\]가 아닌 \[x, y, 1\]을 사용한다. `동차(homogeneous) 좌표계`라는 개념으로 머신러닝에서 흔하게 사용하는 테크닉이다. 본론으로 돌아와 코드는 아래와 같다.

```cpp
double d_x = 100;
double d_y = 150;
Mat affine_matrix = Mat_<double>(
	{ 2, 3 }, { 1, 0, d_x, 0, 1, d_y }
);
warpAffine(img, dst, affine_matrix, Size());
```

---

## 전단 변환

![shear-x](shear-x.png)

`전단(shear)` 변환은 직사각형을 평행사변형으로 비트는 변환이다. 위 이미지는 x(가로) 방향으로 비튼 모습이다. 아래쪽으로 갈수록, 다시 말해 y 좌표가 증가할수록 변화가 커진다. 즉, x 좌표의 변화는 y에 비례한다.

![explain shearing](shear-explain.png)

$$x' = x + m_x y$$

$$y' = y$$

여기서 $m_x$은 변화 정도를 나타낸다. $m_x$가 클수록 x 방향으로 강하게 비튼다.

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & m_x & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

위 행렬은 x 방향으로 비트는 형태라면, y(세로) 방향으로 비트는 경우를 생각해 보자.

![shear-y](shear-y.png)

$$x' = x$$

$$y' = y + m_y x$$

같은 맥락에서 y 좌표의 변화는 x에 비례한다.

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ m_y & 1 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

```cpp
// shear_x
double m_x = 0.5;
Mat affine_matrix = Mat_<double>(
    { 2, 3 }, { 1, m_x, 0, 0, 1, 0 }
);

int x = img.cols;
int y = img.rows;
Size dst_size = Size(cvRound(x + y * m_x), y);
warpAffine(img, dst, affine_matrix, dst_size);


// shear_y
double m_y = 0.5;
Mat affine_matrix = Mat_<double>(
    { 2, 3 }, { 1, 0, 0, m_y, 1, 0 }
);

int x = img.cols;
int y = img.rows;
Size dst_size = Size(x, cvRound(y + x * m_y));
warpAffine(img, dst, affine_matrix, dst_size);
```

---

## 크기 변환

![resized](resize.png)

`크기(scale)` 변환은 이미지를 확대하거나 축소하는 변환이다. x, y에 확대/축소할 비율을 곱하면 크기가 변한다.

$$ x' = s_x \cdot x $$

$$ y' = s_y \cdot y $$

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

```cpp
double s_x = 0.7;
double s_y = 0.9;

Mat affine_matrix = Mat_<double>(
	{ 2, 3 }, { s_x, 0, 0, 0, s_y, 0 }
);
warpAffine(img, dst, affine_matrix, img.size(), INTER_LINEAR);
```

또는 `resize`를 통해 쉽게 처리할 수 있다.

```cpp
resize(img, dst, Size(), s_x, s_y);
```

여기서 의문이 생긴다. 행렬 크기가 달라진다. 따라서 이미지 픽셀의 개수가 달라진다.

예를 들어, 2 x 2 이미지를 4 x 6 이미지로 늘리려 한다. 기존 이미지는 4개의 픽셀(정보)만 가지고 있지만, 확대한 이미지는 24개의 픽셀을 가진다. 이때 발생한 공백을 채우는 방법이 **보간법(interpolation)**이다.

### 양선형 보간법

대표적으로 `양선형(bilinear) 보간법`이 있다. OpenCV에서 `INTER_LINEAR`이라는 플래그로 표현되며, 기본(default) 설정이다. 양선형 보간법은 주어진 픽셀 간 **거리를 바탕으로 가중 평균**을 계산해 값을 구한다.

예를 들어 2 x 2 이미지를 4 x 3으로 확대해 보자.

![2x2to4x3](inter-linear-1.png)

노란색으로 표시한 $P_{2, 1}$ 값은 다음과 같이 계산한다.

$$P_{2, 1}=\cfrac{P_{1, 1} \cdot 2 + P_{4, 1} \cdot 1}{2 + 1} \approx 23$$

거리를 기반으로 가중치를 계산하고 평균을 구한다. 이미지 픽셀은 정수형이기 때문에 근삿값으로 처리한다.

여기까지가 일반적으로 알려진 양선형 보간법이다. 하지만 OpenCV를 실행해 보면 예상과 다르다.

```python
cv2.resize(mat, (4, 3), interpolation=cv2.INTER_LINEAR)
"""
Input:
[[10, 50]
 [30, 90]]

Output:
[[10 20 40 50]
 [20 33 58 70]
 [30 45 75 90]]
"""
```

좌표를 할당하는 과정에서 차이가 발생하는 것으로 보인다. (출처: [stackoverflow](https://stackoverflow.com/questions/68976813/how-inter-linear-interpolation-in-opencv-resize-work))

![coordinate system](inter-linear-2.png)

가로 행에 4개의 픽셀이 할당되어야 한다. 따라서 같은 거리로 값을 배치하다 보니 $P_{2,1}'$는 $(0.25, 0)$에 위치하게 된다. 이 가정을 바탕으로 $P_{2,1}'$을 계산해 보자.

$P_{2,1}'$와 $P_{1,1}\leftarrow (0, 0)$ 사이의 거리는 0.25이다. $P_{2,1}'$와 $P_{2,1}\leftarrow (1, 0)$ 사이의 거리는 0.75이다. 따라서 가중 평균을 구하면,

$$P_{2,1}'=\cfrac{P_{1,1} \cdot 0.75 + P_{2,1} \cdot 0.25}{0.25 + 0.75}=20$$

중요한 내용은 아니지만, 결과에 작은 차이가 발생할 수 있다.

### 다양한 보간법

양선형 보간법 외에 [여러 보간법](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)을 지원한다. OpenCV에서 사용 가능한 플래그는 다음과 같다.

- `INTER_NEAREST`: nearest neighbor. 상대적으로 빠르지만 품질이 떨어진다.
- `INTER_CUBIC`: bicubic. 상대적으로 느리지만 품질이 좋다.
- `INTER_AREA`: resampling. 이미지 축소에 유리하다.

---

## 회전 변환

![rotated](rotate.png)

`회전(rotation)` 변환은 이미지를 시계 또는 반시계 방향으로 회전하는 변환이다. 먼저 시계 방향(clockwise) 회전에 대해 알아보자. 간단한 이해를 위해 단위 원 $x^2+y^2=1$을 살펴보자. 아래는 단위 원을 그리는 Python 코드다.

```python
theta = np.linspace(0, 2 * np.pi, 400)
x, y = np.cos(theta), np.sin(theta)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y)
```

단위 원은 $[0, 2\pi]$ 범위의 $\theta$에 대한 $P(cos\theta , sin\theta )$의 집합이다. 즉, $cos\theta$는 x축, $sin\theta$는 y축과 관계가 있다.

![explain with unit-circle](rotate-circle.png)

구체적으로 $P(cos30, sin30)$를 찍어보면 $P(1,0)$를 반시계 방향으로 회전한 모습이다. 시계 방향으로 회전한 파란 점은 빨간 점에 대해 x축 대칭이므로 $P(cos30,-sin30)$이다. 구체적인 유도 과정은 [gaussian37](https://gaussian37.github.io/math-la-rotation_matrix/#%ED%9A%8C%EC%A0%84-%EB%B3%80%ED%99%98-%ED%96%89%EB%A0%AC-%EC%9C%A0%EB%8F%84-1)님의 블로그에 잘 정리되어 있다.

결론적으로 시계 방향 회전에 대한 회전 행렬은 아래와 같다.

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} cos(\theta ) & -sin(\theta) & 0 \\ sin(\theta) & cos(\theta) & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

반시계 방향에 대한 회전 행렬은 다음과 같다.

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} cos(\theta ) & sin(\theta) & 0 \\ -sin(\theta) & cos(\theta) & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

시계 방향 회전에 대한 코드는 다음과 같다.

```cpp
double angle = 30;
double radian = angle * CV_PI / 180;

// (0, 0)를 기준으로한 시계 방향 회전
Mat rotation_matrix = Mat_<double>(
	{2, 3}, {cos(radian), -sin(radian), 0, sin(radian), cos(radian), 0}
);
warpAffine(img, dst, rotation_matrix, Size());
```

하지만 OpenCV는 $\theta$에 대한 회전 행렬을 생성하는 `getRotationMatrix2D` 함수를 지원한다.

```cpp
Point2f center(img.cols / 2.f, img.rows / 2.f); // 이미지 중심
double angle = 30;

Mat rotation_matrix = getRotationMatrix2D(center, angle, 1);
warpAffine(img, dst, rotation_matrix, Size());
```

![rotate center](rotate-center.png)

또는 `rotate`를 통해 쉽게 처리할 수 있다. 하지만 90도 단위로 회전한다는 한계가 있다.

```cpp
rotate(img, dst, ROTATE_90_CLOCKWISE);
```

---

## 대칭 변환

![horizontal flip](flip-h.png)

`대칭(reflection)` 변환은 축을 기준으로 이미지를 뒤집는 변환이다. 먼저 y축을 기준으로 대칭인 이미지를 만들어보자.

![explain flip](flip-explain.png)

수평 대칭인 이미지의 y 좌표는 같고, x 좌표의 부호만 변한다.

$$x'=-x$$

$$y'=y$$

하지만 이미지 좌표를 음수로 표현할 수 없다. 이미지 넓이를 $w$라할 때, x 좌표는 $[0, w)$ 범위를 가진다. 따라서 $w$만큼 평행이동 시켜 범위를 맞출 수 있다.

$$x'=-x+(w-1)$$

\-1이 붙은 이유는 프로그래밍 언어에서 좌표가 0부터 시작하기 때문이다. 넓이가 300이라면 실제로는 \[0, 299\] 범위의 인덱스를 가진다. C++에서 변환 행렬을 사용하기 위해 $w$가 아닌 $w-1$만큼 이동해야 범위를 넘지 않는다. 이를 행렬로 정리하면 다음과 같다.

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} -1 & 0 & w-1 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

```cpp
double w = img.cols - 1;
Mat affine_matrix = Mat_<double>(
	{ 2, 3 }, { -1, 0, w, 0, 1, 0 }
);
warpAffine(img, dst, affine_matrix, Size());
```

같은 맥락에서 x축 대칭은 y 좌표의 부호를 바뀐 뒤 범위를 조정해 주면 된다. 높이가 $h$일 때, y 좌표의 범위는 $[0,h)$이다. 따라서 변환 행렬은 아래와 같이 표현된다.

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & -1 & h-1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

```cpp
double h = img.rows - 1;
Mat affine_matrix = Mat_<double>(
	{ 2, 3 }, { 1, 0, 0, 0, -1, h }
);
warpAffine(img, dst, affine_matrix, Size());
```

![vertical flip](flip-v.png)

OpenCV는 `flip`을 통해 쉽게 이미지를 뒤집을 수 있다.

```cpp
filp(img, dst, flipCode=1);
```

3번째 파라미터는 `flipCode`로 회전축을 지정한다.

- flipCode == 0: 상하 대칭
- flipCode > 0: 좌우 대칭
- flipCode < 0: 상하 대칭 + 좌우 대칭

---

## 투시 변환

![perspective transform](perspective.png)

`투시(perspective)` 변환은 네 점을 기준으로 임의의 사각형을 직사각형 형태로 변환한다. 먼저, 변환을 위해 네 점의 좌표가 필요하다. 왼쪽 카드의 네 꼭짓점 좌표를 $p=(x,y)$라고 정의하겠다. 그리고 좌표 $p$가 이동할 최종 좌표도 필요하다. 오른쪽 이미지의 네 꼭짓점 좌표를 $q=(x',y')$라고 하겠다. 결론부터 이야기하면 변환 과정은 아래와 같다.

$$\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} = M_{trans} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

$$M_{q}=M_{coef}\cdot M_{trans}'$$

$M_{q}$ and $M_{coef}$ are given.

$M_{trans}$는 3 x 3 변환 행렬이다. $w$는 이미지를 조정하기 위한 scale factor이다. $M_{q}$는 최종 목표 좌표인 $q$를 담고 있는 행렬이다. $M_{q}$는 변환 행렬 벡터를 담고 있는 8 x 1 크기의 $M_{trans}'$과 8 x 8 행렬 $M_{coef}$로 나타낸다. 여기서 LU-decomposition 등 방법으로 $M_{coef}$를 [분해](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#gaaf9ea5dcc392d5ae04eacb9920b9674c)한 뒤, $M_{trans}'$를 구한다. $M_{trans}'$를 3 x 3 행렬로 매핑하면 변환 행렬 $M_{trans}$를 얻을 수 있다. 참고로 $M_{trans}$ 내 마지막 값은 1로 고정이기 때문에 8 x 1 행렬을 3 x 3으로 매핑하는 것이 가능하다.

위 과정은 OpenCV 기본값으로 지정된 `DECOMP_LU`를 기준으로 한 설명이다. 세부적인 과정은 분해 방법에 따라 달라지기 때문에 큰 흐름만 읽고 넘어가자.

다행히 OpenCV의 `getPerspectiveTransformation`을 통해 쉽게 변환 행렬을 얻을 수 있다.

```cpp
// 카드의 꼭짓점. 순서대로:
// top-left > top-right > bottom-right > bottom-left
Point2f objectPoint[4] = {
	Point2f(10, 141),
	Point2f(212, 29),
	Point2f(486, 273),
	Point2f(268, 477)
};

int dst_w = 150;
int dst_h = 200;
Point2f dstPoint[4] = {
	Point2f(0, 0),
	Point2f(dst_w - 1, 0),
	Point2f(dst_w - 1, dst_h - 1),
	Point2f(0, dst_h - 1)
};

Mat transform_matrix = getPerspectiveTransform(objectPoint, dstPoint, DECOMP_LU);
warpPerspective(img, dst, transform_matrix, Size(dst_w, dst_h));
```
