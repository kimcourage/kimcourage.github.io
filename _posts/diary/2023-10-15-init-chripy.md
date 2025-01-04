---
title: Chripy 블로그 만들기
tags: [blog]
category: 생각 정리
toc: true
math: true 
img_path: /assets/posts/init-chripy/
---

만약 본인이 `Jekyll`을 사용해본 경험이 있다면 [공식 문서](https://chirpy.cotes.page/posts/getting-started/)를 참고하는 게 더 빠를 수 있다. 그런데 `Jekyll`도 잘 모르고 `FrontEnd`도 잘 모르겠다하는 사람은 이 글을 잘 찾아왔다.

지금 보고 있듯이 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)은 정말 깔끔한 Jekyll 테마이다. 하지만 막상 시작하려니 계속 문제가 생겨서 5시간 정도를 허무하게 날렸다. 모드 변경도 안 되고 난리도 아니였다. 심지어 로컬 PC를 사용할 수 없는 상황이라 `Ruby`를 설치하고 build를 할 수 없어 더 골치 아팠다. 같은 문제를 겪지 않을 수 있게 과정을 기록해봤다.

![](test-in-prod.png)

> 처음 글을 작성할 당시는 군 복무 중이었기 때문에 Github 웹 에디터에서 글을 작성하고 바로 build, deploy하는 묘기를 선보였다. 하지만 현재(2025)는 로컬 환경에서 작업하고 있다. 따라서 글도 로컬 환경에서 확인 후 빌드하는 방법으로 수정하였다.
{: .prompt-info }

---

## 세팅된 Repo Fork하기

[Repo](https://github.com/denev6/denev6.github.io)로 가면 이 블로그가 돌아가고 있는 Repo(저장소)가 보인다.

![fork](fork.png)

여기서 상단의 `Fork`를 클릭하자. 그럼 해당 Repo가 자신의 Repo로 복사된다. 복사된 곳에서 본인의 블로그로 수정해 쓸 수 있다.

![create repo](mk-repo.png)

중간에 생략한 부분은 나중에 수정할 수 있으니 중요하지 않다. 중요한 건 딱 하나다.

- Repo 이름은 `{유저 ID}.github.io`로 만들어야 한다.
- `Create Fork`를 누른다.

앞에 보면 자신의 Github ID가 있다. 그걸 똑같이 적고 + `.github.io`를 적으면 된다. 그리고 기다리면 Repo가 그대로 복사된다.

## Github Pages

아직은 Repo만 복사된 상태이다. Github Pages를 해줘야 서버에 올릴 수 있다.

![actions](actions.png)

> Settings > Pages > Build and deployment > Source
{: .prompt-info }

Repo에서 `Github Actions`를 선택하면 `GitHub Pages Jekyll`라는 옵션이 생긴다. 그럼 `Configure` 버튼을 누른다. 그리고 아무런 수정 없이 계속 `Commit Changes`를 누르면 된다. 그런 뒤 `유저.github.io`로 블로그에 들어가보자.

> 유저의 이름이 Denev6라면 블로그 주소는 https://denev6.github.io로 생성된다. 이름에 대문자가 있어도 소문자로 입력해야 한다.
{: .prompt-info }

주소로 들어갔을 때 블로그가 보이면 반은 성공이다. deploy가 완료되면 Repo에 `✅초록 체크`가 생긴다. 안 보인다면 조금 기다렸다 들어가거나 새로고침을 하면 보인다. 완료되었다면 `git clone`으로 로컬 환경으로 저장소를 불러온다.

```bash
git clone https://github.com/이름/이름.github.io.git
```

## 초기화 세팅

지금 이 블로그를 복사해 가져간 것이기 때문에 글도 적혀있고 불필요한 세팅도 되어 있을거다. 일단은 초기화를 해보자. 아래 내용은 반드시 해야하는 작업이다.

- `/assets/posts` 모두 삭제
- `/assets/img` 모두 삭제
- `/_posts` 모두 삭제
- `/_sites` 모두 삭제
- `/scripts` 모두 삭제
- `/_config.yml` 내용 삭제 후, `/tools/_config_init.yml` 내용 복사/ 붙여넣기

> _config.yml을 삭제하지 않으면 블로그 세팅이 초기화되지 않는다. 이 작업은 필수다. tools/_config.yml에 미리 초기화 시켜둔 파일이 있으니 그대로 복붙해서 사용하자.
{: .prompt-warning }

### _config.yml

`_config.yml`은 블로그의 전반적인 세팅을 담당하는 파일이다. `_config.yml`을 초기화했다면 잘 읽어보고 입력하자. 주요 설정 값들은 아래와 같다.

```yml
lang: ko  # 블로그 주요 언어 (영어: en)
timezone: Asia/Seoul  # 사용할 시간대
# 검색: http://www.timezoneconverter.com/cgi-bin/findzone/findzone

title: 제목 # 블로그 이름
tagline: 부제목 # 블로그 부제목
description: >- # 블로그 설명
  "설명"

url: "https://이름.github.io" # 프로필 클릭 시 이동할 주소

github:
  username: github_username  # Github ID

twitter:
  username: twitter_username  # 트위터 ID

social:
  name: 이름  # 본인 이름
  email: ID@mail.com  # 메일 주소
  links:
    # 외부 링크. 처음에 적히는 링크는 글 저작권자의 링크로 사용
    - https://github.com/이름 
    # - https://이름.tistory.com/ 
    # - https://www.instagram.com/이름
    # - https://www.linkedin.com/in/username

theme_mode: # light 또는 dark. 비워두면 자동으로 설정
img_cdn: "https://이름.github.io"  # 이미지 기본 경로
# 현재 블로그 주소로 작성하면 편하다. 

avatar: /assets/img/avatar.jpg  # 프로필 이미지 주소

comments:  # 댓글 관련 설정
  active: # 사용할 서비스 작성
  ...

paginate: 10  # 한 페이지에 보여줄 글 수 
```
{: file="_config.yaml" }

사용하지 않을 정보는 비워두면 된다.

### Favicon

Favicon은 반드시 `assets/img/favicons/`에 위치해야 한다. [real-favicon-generator](https://realfavicongenerator.net/)에서 파비콘을 만든 뒤, 생성된 파일들을 그대로 `assets/img/favicons/`에 넣어주면 끝난다. 자세한 내용은 [공식 문서](https://chirpy.cotes.page/posts/customize-the-favicon/)에도 나와 있다.

### About

`About`으로 들어오면 내용이 그대로 남아있을 거다. `_tabs/about.md`에서 내용을 수정할 수 있다. 설정만 남겨두고 다 지워도 된다.

```yaml
---
icon: fas fa-info-circle
order: 1
---
```
{: file="_tabs/about.md" }

### Contact

사이드바 하단에 보면 `Github`부터 여러 아이콘이 있다. 이곳은 `_data/contact.yml`에서 수정할 수 있다.

```yaml
- type: 링크 종류
  # 아이콘은 fontawesome에서 찾을 수 있다.
  icon: "fa-solid fa-pen-to-square"
  # 해당 링크를 현재 탭에서 보여줄지 여부
  # false면 새로운 탭에서 보여준다.
  noblank: false  
  # 이동할 주소
  url: ""
```
{: file="_data/contact.yml" }

### Authors

```yaml
닉네임:
  name: 이름
  twitter: 트위터 ID
  url: 대표 URL 주소
```
{: file="_data/authors.yml" }

`authors`는 글을 쓸 때 글의 저자를 입력할 수 있게 해준다. 지금은 필자의 아이디로 되어 있으니 본인 걸로 수정하자. 

여기까지 했다면 웬만한 초기화는 완료됐다. 사실상 초기화를 하며 동시에 간단한 커스텀까지 했다. 만약 문제 없이 따라왔다면 local server를 띄워보거나 `git commit`해서 잘 돌아가는지 확인해자.

---

## 글쓰기

기본적으로 `마크다운`으로 글을 작성한다. 기본적인 마크다운 기능 외에 Chirpy에서 사용할 수 있는 기능들을 알아보자. 

### 파일명

글은 `_posts/YYYY-MM-DD-NAME.md`에 작성된다. 파일명 양식은 반드시 지켜야 한다. 만약 `_post` 폴더가 없다면 만들면 된다. 

- ie. `2000-01-01-test-post.md`

### 설정

본문을 설정하기 전 상단에 설정 값을 적어야 한다.

**기본**

```yaml
---
title: 제목
category: 카테고리 # [c1, c2...]
tags: 태그 # [t1, t2...]
---
```

**추가**

```yaml
---
date: YYYY-MM-DD HH:MM:SS +/-TTTT
author: 글쓴이 ID
toc: true # 우측에 인덱스 생성
comments: false
pin: true # Home 상단에 글 고정
---
```

### 링크

```markdown
[텍스트](/ai/2023/01/01-title.html)
```

만약 블로그 내의 글을 연결하고 싶다면 `/`부터 작성한다. 만약 연결하고 싶은 글의 파일명이 `2023-01-01-title.md`이고, `ai`라는 파일에 들어가 있다고 하자. 그럼 `(/ai/2023/01/01/title.html)`로 작성한다.

```markdown
[텍스트](#제목)
```

동일한 글 내의 제목으로 이동하고 싶다면 `#`을 사용한다. 예를 들어 제목이 `모델 소개`이라면 `모델-소개`로 작성한다.

### 수식

```yaml
---
math: true 
---
```

$$ \cfrac{1}{N}\sum_{i=1}^{k}x_i $$

`$$수식$$`으로 Latex 문법의 수식을 작성한다.

만약 인라인으로 $\cfrac{1}{N}$ 이렇게 수식을 작성할 때는 `$수식$`으로 쓴다. 공식문서는 inline도 `$$수식$$`으로 작성하라고 하지만 실제 작성해보니 깨지는 경우가 대부분이었다. 

### 이미지

```yaml
---
img_path: /assets/img/
---
```

이미지 경로를 요약해 쓸 수 있다. 만약 `img_path`가 `/assets/img`이고, 본문에서 `flower.png`를 사용했다면 `assets/img/flower.png`를 불러온다. 

```yaml
---
image:
  path: /path/to/image
  alt: image alternative text
---
```

썸네일 이미지를 지정할 수 있다.

![avatar](avatar100.png)
_아바타 이미지_

```markdown
![alt](/path/to/image)
_Caption_
```

이미지 바로 밑에 `_캡션_`을 작성할 수 있다.

```markdown
![alt](img.png){: w="700" h="400" }
```

이미지 크기를 지정할 수 있다.

```markdown
![alt](img.png){: .normal }
![alt](img.png){: .left }
![alt](img.png){: .right }
```

이미지를 정렬할 수 있다.

```markdown
![Light mode only](/path/to/light-mode.png){: .light }
![Dark mode only](/path/to/dark-mode.png){: .dark }
```

라이트모드와 다크모드에 이미지를 따로 적용할 수 있다.

![avatar](avatar100.png){: .shadow }

```markdown
![alt](img.png){: .shadow }
```

이미지에 그림자 효과를 줄 수 있다. (다크모드에서는 티가 안 날 수도 있다.)

### 프롬프트

```markdown
> 내용
{: .prompt-info }
```

> prompt-info
{: .prompt-info }

> prompt-tip
{: .prompt-tip }

> prompt-warning
{: .prompt-warning }

> prompt-danger
{: .prompt-danger }

### 코드 블럭

```markdown
{: file="path/to/file" }
```

코드 블록 바로 밑에 작성하면 코드의 파일명을 지정할 수 있다.

```go
fmt.Print("Hello World!")
```
{: file="hello.go" }

<br/>

```markdown
{: .nolineno }
```

줄번호를 생략한다.

```go
for i, name := range names {
    b.WriteString(name)
}
```
{: .nolineno }

## 확인 및 빌드

먼저 PC에 **Ruby와 jekyll, bundler가 설치되어 있어야 한다**. 자세한 설치법은 이미 많은 블로그에서 다루고 있기 때문에 넘어가겠다. "ruby jekyll bundler 설치"로 검색하면 많이 나온다.

설치가 완료되면 아래 명령어로 로컬 서버를 실행한다.

```bash
bundle exec jekyll serve
```

이제 브라우저를 열고 "http://127.0.0.1:4000/"로 접속하면 작성한 파일이 보인다. 종료하고 싶을 땐 터미널에서 `Ctrl+C`를 누른 뒤, `y`를 입력하고 엔터하면 된다.

문제가 없다면 github에 올리기 전, build를 해주어야 한다. build 하지 않고 올리면 블로그가 보이지 않는다.

```bash
bundle exec jekyll b -d "_site" --incremental
```

`_site` 폴더 아래에 뭔가가 많이 생겼다면 성공이다. 이제 github에 올리면 나머지는 github actions이 자동으로 처리한다.

```bash
git add .
git commit -m " "
git push
```

> 만약 build하는 과정을 자동화하고 싶다면 .github/workflows/pages-deploy.yml에 주석 처리된 부분을 수정하면 된다.
{: .prompt-info }
