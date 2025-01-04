---
title: 데코레이터, 컨텍스트 매니저로 성능 측정
tags: [Python]
category: Playground
toc: true 
---

알고리즘 또는 패키지 성능을 테스트할 때 `시간`과 `메모리`를 측정할 일이 정말 많다. 그런데 특히 메모리와 관련해 정리된 글을 못 찾았다. 

그래서 시간과 메모리 측정을 위해 사용할 수 있는 방법들을 구상해 정리해보았다. 그리고 `Decorator`+`시간측정`, `Context Manager`+`메모리 측정`를 사용해 파이썬다운 이쁜 코드를 적어보았다. 

---

## Decorator

### 생성

`Decotrator`를 사용하면 함수의 시작과 끝에 특정 동작을 실행할 수 있다. `Decorator`를 생성하기 위해서는 `wrapper`와 `inner` 2종류의 함수를 정의해야 한다. 아래는 데코레이터 함수의 공식이다.

```python
# Decorator 함수
def wrapper(func):
    def inner(...):
        # 시작 코드
        func(...)
        # 종료 코드
        return 
    return inner
```

`wrapper`는 파라미터로 반드시 함수를 받는다. 혼동을 피하기 위해 이 함수를 `함수'`라고 적겠다. 

그럼 `inner`에서 파라미터로 받은 `함수'`를 실행할 수 있다. 이때 `함수'` 앞뒤로 동작을 정의할 수 있다. 

```
wrapper로 함수' 받음 
→ 함수'를 inner로 전달 
→ inner에서 파라미터 입력 받음 
→ inner 함수 실행
```

보다시피 `wrapper`는 `inner`를 실행하기 위해 `함수'`를 받아오는 역할이 전부다. 왜 이렇게 복잡하게 구성하는가 의문이 들 수 있다. 이렇게 하면 `@`로 파이썬 마법을 부릴 수 있다.

### 실행

`Decorator` 함수를 정의한 뒤에 `@wrapper`를 쓰면 일반 함수를 데코레이터가 적용된 함수로 변환해준다. `@wrapper` 밑에 `def`로 정의된 함수가 위에서 봤던 `함수'`이다. 

```python
@wrapper
def f(x): ...
```

이 방법은 함수를 재사용할 수 있게 해준다. 아래 예시를 보자.

### 예시

함수 시작과 끝에 `[Start Point]`, `[End Point]`를 출력해야 하는 상황이다. 여기서 데코레이터를 활용해보자. 

#### 원본

```python
def hi_to(name):
    print("[Start Point]")
    print(f" Hi!!! {name}")
    print("[End Point]")

def hello_to(name):
    print("[Start Point]")
    print(f" Hello!!! {name}")
    print("[End Point]")

# 실행
hi_to("James")
hello_to("James")
```

#### 데코레이터 적용

```python
def print_points(func): # wrapper
    def inner(name):
        print("[Start Point]")
        func(name) # 여기에 쓸 함수를 정의
        print("[End Point]")
    return inner

@print_points
def hi_to(name):
    print(f" Hi!!! {name}")

@print_points
def hello_to(name):
    print(f" Hello!!! {name}")

# 실행
hi_to("James")
hello_to("James")
```

```
[Start Point]
 Hi!!! James
[End Point]
[Start Point]
 Hello!!! James
[End Point]
```

만약 데코레이터를 사용하지 않는다면 매번 `print(...)`를 해야 한다. 하지만 데코레이터를 활용하면 한 번만 작성해도 된다. 출력 구문이 바뀐다해도 한 번의 수정으로 모두 적용할 수 있다. 코드를 재사용하기 위해 변수나 함수를 사용하는 것과 같은 맥락이다.

또 `hi_to`와 `hello_to` 함수 정의를 보면 어떤 동작을 하는지 쉽게 이해할 수 있다. 반복되는 동작을 함수 안에 다 정의하지 않으니 코드의 가독성이 높아진다. 

---

## 시간 측정

### time 모듈

시간을 측정하고 싶다면 `process_time`과 `perf_counter`가 있다. 

```python
import time

# 순수 연산 시간
start = time.process_time()
# 코드...
end = time.process_time()
print(f"수행 시간: {end - start}초")

# 전체 소요 시간
start = time.perf_counter()
# 코드...
end = time.perf_counter()
print(f"수행 시간: {end - start}초")
```

- `process_time`: CPU에서 sleep, io 등 pending 시간을 제외하고 측정한다. 순수 연산 시간만을 측정한다. 
- `perf_counter`: 연산에 사용된 모든 시간을 측정한다. 

둘은 명확한 차이가 있기 때문에 측정하는 목적에 따라 선택해 사용할 수 있다. 

### 데코레이터 적용

```python
import time

# Decorator 함수 정의
def with_timer(func):
    def timer(*args, **kwargs):
        """Returns:
            - any: func의 반환값
            - float: func의 실행 시간 (초)
        """
        start = time.process_time()
        # func 실행
        retval = func(*args, **kwargs)
        end = time.process_time()
        duration = end - start
        return retval, duration
    return timer

# Decorator 적용
@with_timer
def test():
    zeros = [0 for i in range(10**8)]
    return len(zeros)

#실행
ret, sec = test()
print(f"test -> {ret}: {sec:.3f}s")
```

```
test -> 100000000: 3.658s
```

시간을 측정할 함수를 정의할 때 `@with_timer`를 붙여 사용할 수 있다. 함수에 `@타이머와 함께`라고 써주니 가독성도 좋다.

---

## Context Manager

`Context manager`는 `with` 구문을 통해 시작과 끝 동작을 정의할 수 있는 기능이다. 파이썬다운 코드를 작성할 수 있는 유용한 기능이다.

대표적으로 `open` 구문이 있다.

```python
with open("file.txt", "r") as f:
    f.read()
```

### 정의

`Context manager`는 객체로 정의된다.

```python
class Context(object):
    def __enter__(self):
        # 사전 작업
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # 사후 작업
```

- `__enter__`: `with`문 시작 전에 실행할 동작. 반환값은 `as`를 통해 받을 수 있다.
- `__exit__`: `with`문을 닫으며 실행할 동작. 파라미터로 오류와 관련된 정보를 받는다. 

주의할 점은 `__exit__`가 `True`를 반환하면 예외가 발생해도 문제 없이 코드를 진행한다.

```python
class Context(object):
    def __exit__(self, exc_type, exc_value, traceback):
        return True
```

### 실행

```python
with Context() as context:
    # 필요한 처리 (들여쓰기)
```

`with`를 통해 정의한 컨텍스트를 사용하고, `as`를 통해 `__enter__`의 반환값을 받아온다. 들여쓰기를 통해 컨텍스트 내에서 작동할 코드를 정의한다. 들여쓰기가 끝나는 지점에서 `__exit__`가 실행된다. 

풀어쓰면 아래와 같다.

```python
class Context(object): ...

ct = Context()
context = ct.__enter__()
# with문에서 들여쓰기한 코드
ct.__exit__()
```
### 변형

`Context Manager`를 정의하는 다양한 방법이 있다. 굳이 알 필요는 없지만 궁금하면 가볍게 살펴보자. 

#### ContextDecorator

데코레이터를 객체로 정의하는 방법이다. 

```python
import contextlib

class Context(contextlib.ContextDecorator):
    def __enter__(self): ...

    def __exit__(
            self, 
            exc_type, 
            exc_value, 
            traceback
        ): ...
```

`contextDecorator`를 상속받은 객체는 `Decorator`로 활용 가능하다.

```python
@Context
def func(): ...
```

위에서 봤던 데코레이터와 동일하게 작동한다. 다만 데코레이터 함수를 정의하지 않고 객체의 형태로 정의한 거다. 이 방법은 `with...as`와 달리 컨텍스트 객체 자체를 받아와 직접 제어할 수는 없다. 

#### contextmanager

이번에는 함수로 정의하고, `with`로 실행하는 방법이다.

```python
import contextlib

@contextlib.contextmanager
def context():
    # 사전 작업: __enter__과 같은 역할
    yield 반환값
    # 사후 작업: __exit__과 같은 역할
```

```python
with context() as 반환값:
    # 필요한 작업
```

`@contextmanager` 함수에서 작업을 한 후 `yield` 키워드로 대기한다. 그 동안 `with` 구문 내 서브루틴이 실행되는 형태이다. 정의하는 방식이 데코레이터의 `inner` 함수와 유사하다. 반면, 사용할 때는 `with...as` 구문을 사용하는 혼종이다. 

---

## 메모리 측정

### 메모리 확인

```python
import os
import psutil

pid = os.getpid()
process = psutil.Process(pid)
memory = process.memory_info().rss

print(f"사용 중인 메모리: {memory / 1024**2}MiB")
```

현재 할당된 `pid(process id)`를 찾아 `Process` 객체를 만든다. 그리고 `memory_info`를 통해 메모리 사용량을 가져올 수 있다.

시간 측정하듯 (종료 시점 메모리) - (시작 시점 메모리)를 하기에는 `GC(Garbage Collector)`로 정리된 메모리나 함수 호출이 종료되면서 사라진 값 등을 측정할 수 없다. 따라서 계속 추적해가며 메모리를 확인하기로 했다. 

### 추적

```python
import sys
  
def tracer(frame, event, arg):
    # 필요한 작업 수행
    return tracer
   
sys.settrace(tracer)
# 추적할 코드
sys.settrace(None)
```

`settrace`는 `tracer` 함수를 계속 추적할 수 있도록 해준다. `tracer`는 3개의 파라미터를 받아야한다.

- `frame`: 현재의 스택 프레임
- `event`: 'call', 'line', 'return', 'exception', 'opcode' 중 하나이다.
- `arg`: [문서 참고](https://docs.python.org/3/library/sys.html#sys.settrace)

마지막으로 `settrace(None)`으로 추적을 종료한다. 

메모리 추적에 필요한 키워드만 뽑아왔으니 예시를 보자.

#### 예시

```python
import sys

def my_tracer(frame, event, arg):
    """tracer 함수"""
    # {발생한 이벤트}, {실행된 함수명} 출력
    print(f"{event}\t{frame.f_code.co_name}")
    return my_tracer

def test():
    """테스트를 위한 함수"""
    list_ = [i for i in range(2)]
    print(" --출력:", list_)
    return list_

print("event\tfunction")
print("-----------------")

# 추적 시작
sys.settrace(my_tracer)
_ = test()

# 추적 종료
sys.settrace(None)
```

```
event   function
-----------------
call    test
line    test
call    <listcomp>
line    <listcomp>
line    <listcomp>
line    <listcomp>
return  <listcomp>
line    test
 --출력: [0, 1]
return  test
```

`test`가 호출(call)되고 `test` 내부에 있던 `list-comprehension`이 실행되는 과정을 모두 추적했다. 

### 컨텍스트 적용

```python
import os
import sys
import psutil

class Tracer(object):
    """Params:
        - max_record (int): 예상되는 동작 수
        - *to_trace (...str): 추척할 동작 이름 
       Attr:
        - record (list[int]): 기록된 메모리 사용량
    """
    
    def __init__(self, max_record, *to_trace):
        self._to_trace = to_trace
        self.__process = psutil.Process(os.getpid())
        self.__max_record = max_record
        self.__record = [0 for _ in range(self.__max_record)]
        self.__count = 0

    def __enter__(self):
        """with문 추적 시작"""
        sys.settrace(self.trace)
        return self

    def __exit__(self, *args):
        """with문 추적 종료"""
        sys.settrace(None)

    def trace(self, frame, event, arg):
        if self.__count >= self.__max_record:
            # 예외 처리
            messages = [
                "예상된 동작보다 많은 동작이 실행되었습니다.",
                f"max_record를 {self.__max_record}보다 크게 설정해주세요."
                "추적을 종료합니다."
            ]
            print("\n".join(messages))
            self.__exit__()
            return
            
        if (frame.f_code.co_name in self._to_trace) and (
            event in ("call", "line", "return")
        ):
            # 추적한 메모리 기록
            self.__record[self.__count] = self.__process.memory_info().rss
            self.__count += 1
        return self.trace

    @property
    def record(self):
        """추적된 메모리 반환"""
        return self.__record[:self.__count]
```

```python
with Tracer(max_record, *to_trace) as tracer:
    # 함수 실행
    
memory = tracer.record  # 메모리 기록 (list)
```

객체는 복잡해 보이지만 사용법은 간단하다. 예상되는 동작의 수와 추적할 함수 이름만 전달해주면 된다.

`Tracer`라는 컨텍스트를 정의해 시작부터 끝까지 `sys.settrace`로 추적한다. `trace`를 보면 실행된 함수의 이벤트가 `call` · `line` · `return`일 때 메모리를 저장한다. 

> 이 코드도 불완전하다. 파이썬 리스트는 동적으로 값을 추가한다. 심지어 growth-factor가 1.125로 작다. 따라서 추적 정보 기록 시 리스트로 인한 메모리 증가가 발생할 가능성이 매우 높다. 이렇게 되면 함수 때문에 메모리가 증가하였는지, 리스트가 할당되며 증가하였는지 알 수 없다. 따라서 처음부터 일정 길이의 리스트를 생성한 뒤, 리스트 내 값을 수정하는 방식으로 제작하였다. 지금으로서는 Tracer를 선언할 때, 예상되는 동작보다 넉넉하게 잡는 것이 중요하다. 
{: .prompt-warning} 

#### 실행

```python
import matplotlib.pyplot as plt

def temp():
    """테스트를 위해 만든 함수"""
    a = [i for i in range(10 ** 5)]
    b = [i for i in range(10 ** 2)]
    del a
    c = [i for i in range(10 ** 4)]
    d = [i for i in range(3)]
    final = b + d
    return final

with Tracer(10, "temp") as tracer:
    # tracer 실행
    temp()

# 메모리 기록 가져오기
record = [x / 1024 ** 2 for x in tracer.record]
# 시각화
plt.figure()
plt.plot(record)
plt.ylabel("Memory (MiB)")
plt.show()
```

![](/assets/posts/others/memory-usage.png)

```
     event   내용
----------------------------------------------
0~1: call    `temp` 호출
1~2: line    `a = [i for i in range(10 ** 5)]`
2~3: line    `b = [i for i in range(10 ** 2)]`
3~4: line    `del a`
4~5: line    `c = [i for i in range(10 ** 4)]`
5~6: line    `d = [i for i in range(3)]`
6~7: line    `final = b + d`
7~8: line    `return final`
8~ : return  `final` 반환
```

처음 시작점을 기준으로 어디서 메모리가 많이 사용되었는지, 왜 메모리가 튀었는지 등 정보를 볼 수 있다. 
