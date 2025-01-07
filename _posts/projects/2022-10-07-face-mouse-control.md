---
title: 실시간 얼굴 인식 프로젝트를 진행하며
tags: [AI, CV, Python]
category: Projects 
toc: true
img_path: /assets/posts/face-control/
pin: true
---

> Github: [/face-mouse-control](https://github.com/denev6/face-mouse-control)  
> 논문: [얼굴 인식과 Pyautogui 마우스 제어 기반의 비접촉식 입력 기법](http://koreascience.or.kr/article/JAKO202228049092231.pdf)

![](demo.png)

```bash
$ git clone https://github.com/denev6/face-mouse-control.git
$ pip install -r requirements.txt
$ python main.py
```

실행할 때 프로젝트 최상단에 있는 `main.py`를 실행한다. `settings-gui.py`는 사용자 개인 맞춤 설정을 도와준다. 

## 동기

외할아버지가 손이 불편하셔서 기계 조작에 어려움이 있으셨다. 나는 "만약 마우스와 키보드 없이 조작할 수 있다면 어떨까?"라는 생각이 들었다. 학교에서 교수-학생 협력(Co-Deep learning) 프로젝트를 진행하고 있었고, 팀을 꾸려 `비접촉 입력 기법`을 연구하기 시작했다.

## 최종 목표

처음부터 상용화 가능한 수준이 목표였다. 일상의 불편함을 해결하는 프로젝트이기 때문에 이론에만 머문다면 의미가 없다. 따라서 정확도는 물론 적은 자원으로 빠른 속도를 내야 한다.

구체적으로 Intel-i5 CPU 위에서 작동해야 한다. 외장 GPU가 없는 노트북이 많다고 생각해 CPU만으로도 잘 작동하는 서비스를 구현하기로 했다. 

## 구현 목표

얼굴 주시방향을 따라 마우스가 이동하며, 눈 깜빡임으로 마우스 좌클릭을 실행한다. 작은 목표로 나누면 아래와 같다.

- 카메라로 얼굴 랜드마크를 인식해야 한다. 
- 랜드마크로 주시방향을 계산해야 한다. 
- 눈 깜빡임을 포착하고 판단해야 한다. 
- 연산된 정보를 이용해 마우스를 조작해야 한다. 

## 얼굴 인식 모델

`FaceMesh`가 가장 준수한 성능을 보였다. `FaceMesh`는 실시간 얼굴 인식에 특화된 모델로 빠른 처리 속도가 특징이다. `HOG`, `SSD` 등 다른 모델과 비교했을 때, 속도와 정확도 모두 뛰어난 모습을 보였다. 

동일한 문제를 계산하는데 소요된 시간이다.

|Model|runtime(s)|
|:-:|:-:|
|FaceMesh|1.12|
|HOG|13.62|
|SSD|3.86|

자세한 분석은 [논문](http://koreascience.or.kr/article/JAKO202228049092231.pdf)의 `3.2.1. 사용한 모델들`에 기록했다.

> FaceMesh 정보는 아래 문서에서 확인할 수 있다.  
> Docs: [MedaiPipe/ Face landmark detection guide](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/)  
> 논문: [Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs](https://arxiv.org/abs/1907.06724)
{: .prompt-info }

## 얼굴 방향 계산

얼굴 방향 계산은 `Perspective-n-Point` 문제이다. `FaceMesh`는 3차원 얼굴 랜드마크 좌표를 반환한다. 따라서 카메라 왜곡이 없다면, 이미지 2차원 좌표와 3차원 좌표로 얼굴 회전벡터를 구할 수 있다.

> The cv::solvePnP() returns the rotation and the translation vectors that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame, using different methods.  
> \- [calib3d solvePnP](https://docs.opencv.org/4.5.5/d5/d1f/calib3d_solvePnP.html)
{: .prompt-info }

## 눈 깜빡임 인식

눈의 가로, 세로 비율을 측정해 눈 깜빡임을 인식했다. 논문: [Real-Time Eye Blink Detection using Facial Landmarks](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)에 따라 가로 비율 대비 세로 비율이 감소하면 눈 깜빡임으로 처리했다. 하지만 눈을 무의식적으로 깜빡이는 경우 클릭으로 처리하면 안 된다. 따라서 일정 프레임 이상 감고 있어야 클릭으로 이어진다.

## 입력 구현

Pyautogui를 이용해 모든 조작을 구현했다. Pyautogui 는 마우스 조작, 키보드 조작이 가능하다. 이를 통해 클릭, 이동, 확대/ 축소 기능을 구현했다.

Tkinter로 사이드바에 확대/ 축소 등 버튼을 구현했다. 논문: ["상지장애인을 위한 시선 인터페이스에서의 객체 확대 및 음성 명령 인터페이스 개발"](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE10585014)에 따르면 웹 화면을 140% 확대했을 때 가장 조작하기 편하다고 한다. 따라서 화면 확대/ 축소 기능을 버튼으로 구현했다.

## 구조 설계

![](process.png)

절차적 프로그래밍으로 제작된 코드들을 OOP로 리팩토링하였다. 팀원들은 각자 파트를 나눠 코드를 구현했다. 완성한 코드를 종합해보니 절차적으로 구성되어 결과 코드가 길어졌다. 따라서 각 기능을 객체로 분리했다. 데이터는 객체 속에서 조작되며 다른 객체의 영향을 받지 않는다. 덕분에 역할과 책임이 분명하고 수정이 편하다.

전체 프로세스는 선형으로 설계했다. Python은 뮤텍스인 GIL 때문에 비동기 처리 및 병렬처리에 불리하다. 따라서 각 객체가 순서대로 연산을 진행하는 선형적인 구조가 만들어졌다. (구현 당시 3.8버전을 사용했다.)

## 사용자 피드백

다양한 연령대의 사용자 10명을 모집해 간단한 과제를 해결하게 했다. 예를 들어, 구글 검색(클릭), 웹툰 시청(스크롤 및 확대) 등이 있다. 완료한 참가자는 불편하거나 개선할 점을 설문지에 작성했다.

주요 내용은 마우스가 바로 정지하지 않는 문제였다. 사용자가 마우스를 움직이기 위해 좌/우측을 주시한다. 이후 정지하기 위해 순간 정면을 바라본다. 이때 고개를 정면으로 돌리는 순간에도 여전히 좌/우측을 바라보기 때문에 마우스가 계속 이동한다. 즉, 마우스가 바로 정지하지 않고 미끄러진다는 것이다. 

![](stop-sim.png)
_정지하는 상황 재연_

해결책으로 변화률 계산을 추가했다. 순간 빠르게 정면을 주시할 경우 즉시 마우스를 정지한다. 다시 말해, 각도의 급격한 감소를 발견하면 즉시 정지 명령을 실행한다.

그외 수정 사항은 사이드바 버튼 배열 개선, 처리 속도 측정을 이용한 프레임 드랍 감소 등이 있다.

## 전문가 피드백

감사하게도 학교 도움으로 현업 전문가분들에게 피드백 받는 기회를 얻었다. 우리 팀이 프로젝트에 대해 발표하고 전문가분과 질의응답하는 시간을 가졌다. 피드백은 아래 사진으로 첨부했다.

> Palete팀의 얼굴 인식 활용 대체 입력 프로젝트가 매우 잘 수행되었습니다. 특히, 기존의 연구나 상품들이 제공하지 못했던 여러 기법들을 머신러닝과 딥러닝 기술을 잘 활용해 저렴한 비용으로 사용 가능하도록 새로운 방식을 잘 제안하였다고 판단됩니다. 시간과 비용의 관점에서 기존의 Pre-train된 머신러닝 모델들을 잘 활용하고, dib과 OpenCV 패키지들 잘 활용하였습니다. 한걸음 더 나아가, 이후에는 직접 여러 face landmark recognition 모델을 활용해 인식률을 높이고, 이후 피드백을 받은 부분을 조금 더 보완해 나간다면, 상품화와 실제 서비스로 내 놓아도 손색이 없을 정도로 훌륭한 서비스가 될 것으로 예상됩니다. 끝으로, 이런 과정과 진행을 모두 github repo에 공개해 이후에도 지속적인 발전이 가능한 오픈소스로 꾸준하고 지속적인 관심을 받을 것으로 기대됩니다. - Microsoft 김대우 이사

![전문가 의견: Microsoft 김대우 이사](advise.png)

## 느낀 점

막연하게 머리 속으로 생각해왔던 기능을 구현해 보고 동작하는 모습을 보니 뿌듯했다. 실시간 영상 정보 처리 방법이나 얼굴 객체 검출 등 작업을 수행하며 실력이 발전했다. 무엇보다 팀 단위로 작업한 프로젝트이기 때문에 *'[it works on my computer](https://donthitsave.com/comic/2016/07/15/it-works-on-my-computer)'* 문제가 발생하지 않기 위해 환경 구축으로 버전을 통일하고, Github으로 버전을 관리하는 등 협업 경험을 쌓을 수 있었다. 생각보다 블로그가 도움이 되었다. [Conda](https://denev6.tistory.com/entry/Anaconda3)나 [OpenCV](https://denev6.tistory.com/entry/cam-capture) 글도 프로젝트 당시 팀원에게 공유하기 위해 작성한 글이다. 알고 있는 내용을 구구절절 설명하는 것보다 블로그 링크를 공유하는 편이 효율적이고 편리했다. 다음에 실시간 이미지 처리를 하게 되면 이번에 배운 내용이 도움이 될 것 같다.
