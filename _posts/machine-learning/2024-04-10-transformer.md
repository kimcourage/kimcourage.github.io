---
title: Attention is all you need
category: Machine-Learning
toc: true 
math: true
img_path: /assets/posts/transformer/
---

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

본 글은 "**_Attention is All You Need_**" 논문을 번역 및 분석했다. 일부 문장은 맥락에 따라 의역되었으며, 명확한 이해를 위해 부분적으로 설명을 추가했다. 주요 용어는 정확한 의미 전달을 위해 영문 그대로 작성했다. (예: recurrent, convolutional 등)

## Abstract 

기존 시퀀스 모델은 encoder-decoder가 포함된 복잡한 recurrent나 convolutional 신경망을 기반으로 한다. 본 논문은 recurrence와 convolution 없이 **attention mechanisms을 기반으로 하는 간단한 Transformer 구조를 제안**한다. 2종류의 기계번역 문제에서 좋은 성과를 보였고, 병렬화를 통해 학습 시간을 단축했다. 본 모델은 WMT 2014 영어-독일어 번역에서 28.4 BLEU를 달성했다. WMT 2014 영어-프랑스어 번역은 41.8 BLEU로 단일 모델 SOTA를 달성했다. 8개 GPU로 3.5일을 학습했다. Transformer를 영어 문장 성분 파싱에 적용했고, 다른 문제에도 적용 가능하다는 사실을 확인했다. 학습 데이터가 큰 상황과 제한된 상황에서 모두 잘 학습되었다.


## Introduction

RNN, LSTM, GRU는 기계 번역이나 언어 모델 분야에서 준수한 성능으로 입지를 확고히 해왔다. Recurrent 모델은 $t$ 시점 hidden state인 $h_t$를 학습하기 위해 $h_{t-1}$를 사용한다. 이러한 순차적인 구조는 병렬 연산을 활용할 수 없어 긴 시퀀스에 치명적이다. 최근 factorization tricks나 conditional computation을 이용해 연산 효율과 앞서 말한 문제를 개선했다. 하지만 여전히 모델 구조에 따른 근본적인 제약이 있다.

Attention mechanisms는 시퀀스 길이에 관계없이 의존성 모델링이 가능하며, 다양한 문제에서 좋은 모습을 보여준다. 하지만 대부분 Attention은 recurrent 구조와 함께 사용된다. 본 논문은 Transformer를 제안하고, recurrence를 피하는 대신 **완전히 attention 구조에 의존하는 방식**으로 입출력 사이 global dependency를 도출한다. Transformer는 병렬 처리를 통해 변역 문제에서 SOTA를 달성했고, 8개의 P100 GPU로 12시간을 학습했다.

## Model Architecture

![](fig1.png)
_fig1_

\*그림에서 좌측이 Encoder, 우측이 Decoder 구조다.

### Encoder-Decoder stacks

**Encoder**는 N=6개의 동일한 층이 연결된 모습이다. 각 층은 multi-head self-attention과 간단한 position-wise fully connected feed-forward network로 구성된다. 각 sub-layer에 대해 residual connection을 적용하고, 뒤이어 정규화를 진행한다. \* 그림에서  residual connection은 multi-head attention 입력을 출력과 합치는 부분을 말한다. (Add)

$LayerNorm(x+Sublayer(x))$

residual connection을 쉽게 처리하기 위해 embedding을 포함한 모든 출력은 $d_{model}=512$ 차원으로 고정한다.

**Decoder**도 N=6개의 동일한 층으로 구성된다. 내부는 Encoder에 한 개 sub-layer을 추가한 형태로, 총 3개 층으로 구성된다. 추가된 층은 Encoder 출력을 받아 multi-head attention을 수행한다. Decoder도 Encoder와 마찬가지로 residual connection과 정규화를 적용한다. 또한 첫 self-attention에 masking을 적용해 output embedding을 상쇄한다. 이를 통해 i번째 위치의 값은 i 이전 값에만 영향을 받도록 한다. \*masking에 대해 다음 챕터(Applications of Attention in our Model)에서 자세히 설명한다.

### Attention

![](fig2.png)
_fig2_

#### Scaled Dot-Product Attention

입력은 $d_k$ 차원의 query, key와 $d_v$ 차원의 value이다. Query와 Key를 점곱한 뒤 $\sqrt{d_k}$로 나누고 softmax를 통해 각 value에 대한 가중치를 얻는다.

$Attention(Q,K,V)=softmax(\cfrac{QK^T}{\sqrt{d_k}})V$

Dot-product attention은 최적화된 행렬 연산 코드를 이용하기 때문에 다른 attetion에 비해 빠르고 공간 효율성이 좋다. 그리고 $d_k$가 큰 값이면 점곱의 결과가 커진다. 이는 softmax 연산 시 매우 작은 gradient로 이어질 수 있다. 문제를 해결하기 위해 $\cfrac{1}{\sqrt{d_k}}$로 스케일링했다.

#### Multi-Head Attention

각 query, key, value에 대해 attention 연산하는 것보다 각각 $d_k,d_k,d_v$ 차원으로 h번 linearly project 하는 것이 효율적이다 (Fig2 참고). 그리고 각 query, key, value에 대해 병렬로 attention을 수행해 $d_v$ 차원의 출력을 계산한다. 이 값은 다시 concat & project 되어 출력이 된다. Mutil-head attention은 다른 위치에 다른 영역에서 온 정보를 한 번에 확인할 수 있다.

$MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O$

where $head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$

$W_i^Q,W_i^K,W_i^V$과 $W^O\in\mathbb{R}^{hd_v\times d_{model}}$는 project 될 때 사용하는 parameter다.

본 연구는 h=8개의 병렬된 attention 층을 사용한다. 각 차원은 $d_k=d_v=d_{model}/h=64$이다. 줄어든 차원 덕분에 전체 연산 비용은 singe-head attention과 유사하다.

\*여기서 차원이 줄었다는 표현은 병렬 연산을 하며 나타난 효과다. 위 설명에 따르면 $d_k=64$차원의 모델 8개를 병렬로 처리한다. 이는 $d_{model}=512$차원의 모델 하나를 처리하는 것과 같다 (512 = 64 x 8). 

#### Applications of Attention in our Model

Transformer는 multi-head attention을 3가지 방식으로 활용한다. 

encoder-decoder attention 층에서 query는 이전 decoder에서 오고, key와 value는 encoder 출력에서 나온다. 따라서 decoder가 모든 입력 시퀀스 위치에 적용된다. 이는 seq2seq에서 전형적인 encoder-decoder attention 구조와 동일하다.

encoder는 **self-attention** 층을 가지고 있다. self-attention에서 key, value, query는 같은 곳에서 나오며, 본 연구에서는 encoder 이전 층의 출력을 말한다. 따라서 encoder 위치가 이전 encoder의 모든 위치를 참고하게 된다. \*자세히 말하면, embedding 된 단어를 key, value, query로 사용한다. 이를 통해 각 벡터 간 거리를 계산한다.

decoder도 마찬가지로 self-attention을 통해 모든 위치를 참조한다. 하지만 auto-regressive 속성을 유지하기 위해 다음 출력의 영향을 받으면 안 된다. 따라서 softmax 입력을 모두 **masking**(-∞로 설정)하는 방식을 scaled dot-production attention에 적용했다.

\*Transformer는 순차적으로 정보를 입력하는 encoder-decoder와 달리 모든 값을 한 번에 입력한다. 따라서 미래 정보를 확인할 수 있다. 예를 들어, "the song _Attention_ by _Newjeans_"라는 문장이 있다고 하자. Newjeans는 Attention 뒤에 위치한다. 따라서 시간 상 Attention → Newjeans 관계를 파악하는 것은 바람직하다. 하지만 Newjeans → Attention 순서로 맥락을 파악하는 것은 바람직하지 않은(illegal) 연결이다. 이러한 문제를 해결하기 위해 masking을 사용한다. masking 된 정보를 -∞로 설정하는 이유는 softmax를 거쳤을 때 0이 되도록 하기 위함이다.

![masked matrix](masking.png)

### Position-wise Feed-Forward Networks

각 sub-layer는 fully connected feed-forward network를 가진다. 모두 동일한 형태로 각 위치에 적용된다. 2개의 linear 층이며 ReLU를 활성화 함수로 사용한다.

$FFN(x)=ReLU(xW_1+b_1)W_2+b_2$

선형 변환에서 각 층마다 다른 파라미터를 가진다. 입출력은 $d_{model}=512$ 차원으로 내부 층은 $d_{ff}=2048$ 차원이다. (2048 =512 x 4. W1, W2, b1, b2에 대해)

### Embeddings and Softmax

$d_{model}$ 차원의 벡터로 입출력을 변환하기 위해 학습된 embedding을 사용한다. decoder 출력을 확률로 변환하기 위해 선형 변환과 softmax를 사용했다. 본 연구는 두 embedding 층과 softmax 이전 선형 변환에서 같은 가중치 행렬을 사용했다. 그리고 embedding 층에서는 가중치에 $\sqrt{d_{model}}$을 곱한다. 

### Positional Encoding

본 모델은 순환 구조가 없기 때문에 시퀀스 순서를 이해하기 위해 토큰에 위치 정보가 필요하다. 이를 위해 **positional encoding**을 encoder와 decoder의 input embedding 밑부분에 추가했다. positional encoding은 embedding과 더해질 수 있도록 같은 $d_{model}=512$ 차원을 가진다. 본 연구는 여러 방법 중 다른 주기를 가지는 sine과 cosine 함수를 이용한다. 

$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$

$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$

*pos*는 위치이며, *i*는 차원이다. 각 차원은 정현파(sinusoid)에 대응된다. 주기는 $2\pi$에서 $10000\cdot 2\pi$가 된다. 

**\*positional encoding 추가 설명**

![positional encoding](pos.png)

말 그대로 embedding 된 단어에 위치 정보를 추가해 주는 역할이다. 위치를 표기하는 방법은 다양하다. 예를 들어 첫 번째 단어는 1 ... i번째 단어는 i로 나타낼 수 있다. 그런데 i 값이 너무 커지면 더했을 때 임베딩된 벡터와 관계없이 아주 큰 값이 된다. 임베딩 벡터는 단어 정보를 담고 있기 때문에 중요하다. 따라서 항상 -1 ~ 1 사이 범위를 가지는 sine, cosine 함수를 선택했다. 

하지만 sine, cosine은 일정한 주기를 가지기 때문에 i가 커지면 중복 값이 발생할 수 있다. 따라서 논문에서는 i마다 다른 주기를 가지도록 PE 함수를 정의했다. 물론 i 값이 매우 커지면 겹치는 경우가 발생할 수 있지만, 현재 연구에서는 i에 비해 주기가 충분히 크기 때문에 문제가 되지 않는다.

![sine and cosine](sin-cos.png)

이를 통해 값이 너무 작거나 크지 않으면서 값이 중복되지 않도록 positional encoding을 수행했다.

## Why Self-Attention?

self-attention을 사용한 이유는 크게 3가지이다.

첫 번째는 **연산 복잡도**가 작다. 다른 하나는 연산을 **병렬로 처리**할 수 있다. 세 번째는 **장거리 의존성**(long-range dependencies)이다. 장거리 의존성을 가지는 학습에서 중요한 요인은 앞뒤로 정보를 주고받을 수 있는 경로의 거리다. 이 거리가 짧을수록 장거리 의존성을 학습하기 쉽다. 그래서 layer 종류에 따라 입력과 출력 간 경로 최대 길이를 비교했다.

![](table1.png)
_table1_

표에서 볼 수 있듯이 self-attention은 상수 시간으로 모든 위치를 연결한다. 반면 recurrent 모델은 O(n)이 걸린다. 단일 convolution에서 kernel 크기 k가 n보다 작으면 모든 입출력 위치를 연결할 수 없다. 따라서 contigious kernel에 대해 $O(n/k)$개의 convolutional 층이 필요하고, dilated convolution에 대해 $O(\log_k(n))$가 들어 오히려 최대 길이가 증가한다. convolutional 층은 k 때문에 일반적으로 recurrent 층보다 비용이 많이 든다. seperable convolutional 층은 복잡도를 $O(knd+nd^2)$으로 매우 크게 줄여주지만 k = n이더라도 self-attetion + feed-forward layer와 동일하다.

추가로 self-attention은 더 많은 해석 가능한(interpretable) 모델을 생산해 낼 수 있다. attention distribution을 살펴보면 아래 그림과 같다.

![](fig5.png)
_fig5_

다양한 문제를 잘 해결할 뿐만 아니라 문장 의미와 문법을 잘 나타낸다.

## Training

### Training Data and Batching

450만 개 문장 쌍으로 구성된 **stardard WMT 2014** 영어-독일어 데이터를 학습했다. 영어-프랑스 번역 문제에서 3600만 개 WMT 영어-프랑스어 데이터를 사용했고, 토큰을 32000 word-piece 단어로 나눴다. 문장 쌍은 시퀀스 길이 정도로 batch 했다. 각 training batch는 대략 25000 source token과 25000개 target token을 담고 있는 문장 쌍이 들어있다. 

### Hardware and Schedule

8개의 NVIDIA P100 GPU로 학습했다. 논문에서 설명한 base model은 각 step이 0.4초 정도로 총 100,000 step, 12시간을 학습했다. big model은 각 step 당 1.0초로 300,000 step, 3.5일을 학습했다.

### Optimizer

**Adam** optimizer를 사용했고, $\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$이다. 아래 수식을 이용해 learning rate를 변화해 가며 학습했다.

$lrate=d^{-0.5}_{model}\cdot min(step\\_num^{-0.5},step\\_num\cdot warmup\\_steps^{-1.5})$

![learning rate](lr.png)

### Regularization

**Residual dropout**: 각 sub-layer가 입력과 더해지고 정규화되기 전에 dropout 시킨다. 추가로 embedding과 positional encoding 합에도 dropout을 적용한다. base model은 $P_{drop}=0.1$을 적용한다.

**Label smoothing**: label smoothing factor로 $\epsilon_{ls}=0.1$을 사용한다. 모델을 모호하게 학습해 perplexity를 해치지만 accuracy와 BLEU 점수를 높여준다. 

## Result

### Machine Translation

WMT 2014 영어-독일어 번역 문제(task)에서 big transformer가 **28.4 BLEU**로 이전에 나온 모델을 능가하는 성능을 보였다. 모델 설정은 Table3에 기록했다. 학습은 8개 P100 GPU로 3.5일이 걸렸다. 심지어 base 모델도 학습 비용 측면에서 이전 모델 성능을 뛰어넘었다.

WMT 2014 영어-프랑스어 번역 문제에서 big model은 **41.0 BLEU**로 이전에 발표된 single model을 뛰어넘었다. 이는 이전 SOTA 모델 학습 비용의 1/4로 달성했다. 영어-프랑스어 번역에 사용한 big model은 dropout rate를 0.1 대신 0.3으로 사용했다.

base model은 마지막 5개 체크 포인트의 평균으로 구했으며, 각 체크 포인트는 10분 간격으로 나왔다. big model은 마지막 20개 체크 포인트를 평균 내 사용했다. beam search를 사용했고 beam size는 4, length penalty $\alpha$는 0.6이다. 하이퍼파라미터는 validation set을 통해 나온 결과로 결정했다. inference에서 최대 출력 길이를 입력 길이 + 50으로 뒀지만, 가능한 빨리 끝내는 게 좋다.

![](table2.png)
_table2_

위 표는 결과를 요약하며 번역 성능과 학습 비용을 비교한다. 학습 시간, 사용한 GPU 개수, GPU의 single-precision floating-point 성능을 고려해 floating point operations를 예측했다.

### Model Variations

Transformer의 component 별 중요도를 평가하기 위해 base model을 다양하게 변형해 영어-독일어 번역 성능을 측정했다. 앞서 설명했듯 beam search를 사용했고, 대신 체크포인트를 평균 내는 방식은 사용하지 않았다. 결과는 아래 표에서 볼 수 있다. 

![](table3.png)
_table3_

Table3 (A) 열에서 attention head 개수, key-value 차원을 다르게 하되 연산 일관성을 유지했다. single-head attention은 0.9 BLEU로 성능이 하락했고, 너무 많은 head는 성능을 떨어뜨린다.

Table3 (B) 열에서 attention key 차원을 줄이니 성능에 문제가 발생했다. (C)와 (D) 열은 큰 모델일수록 성능이 좋고, dropout이 over-fitting을 막는데 도움이 된다는 사실을 보여준다. (E) 열은 sinusiudal positional encoding 대신 학습된 positional embeddings을 사용했고 base model과 거의 비슷한 결과를 보였다. 

## Conclusion

Transformer는 오직 attention만을 사용한 첫 sequence transduction model이며, 가장 흔하게 사용되는 recurrent 층을 multi-headed self-attention으로 대체했다. 변역 문제에서 recurrent나 convolutional 모델에 비해 훨씬 빠르게 학습한다. WMT 2014 영어-독일어와 영어-프랑스어 문제에서 SOTA를 달성했다.

학습과 평가에 사용한 코드는 [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)에서 확인할 수 있다.