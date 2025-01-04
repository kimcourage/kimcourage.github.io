---
title: PySet을 Go답게
tags: [Go, Python]
category: Playground
toc: true
---

`PySet`은 아주 유용한 자료구조이다. 이를 `Go`로 Go스럽게 구현하기 위해 `CPython`의 소스코드와 `golang` 소스코드를 살펴보았다. `set`과 `map`이 뒤에서 어떻게 작동하는지를 살펴보고 가장 합리적인 방법으로 집합을 구현해보려 한다. 

---

## 문제

`Python`에는 `집합`이라는 아주 유용한 구조가 있다. `set` 객체는 크게 2가지 역할이 있는데, 중복 값을 제거하는 것과 빠르게 값을 탐색하는 것이다. 

```python
n = [1, 3, 3, 5, 6, 3, 8]
n = set(n)
print(n)
# {1, 3, 5, 6, 8}

has_three = (3 in n)
print(has_three)
# True
```

하지만 `Go`는 `set`을 제공하지 않는다. 따라서 `set`과 유사하게 작동하는 객체를 만들어보려 한다. 그런데 `Go`를 곁들인.

---

## CPython의 Set 분석

`Python`의 구현체인 `C`코드를 뜯어보자. 코드는 [Github-python](https://github.com/python/cpython/blob/main/Include/cpython/setobject.h)에 공개되어 있다.

```c
typedef struct {
    PyObject *key;
    Py_hash_t hash;             /* Cached hash code of the key */
} setentry;

typedef struct {
    PyObject_HEAD
    Py_ssize_t fill;            /* Number active and dummy entries*/
    Py_ssize_t used;            /* Number active entries */
    Py_ssize_t mask;
    setentry *table;
    Py_hash_t hash;             /* Only used by frozenset objects */
    Py_ssize_t finger;          /* Search finger for pop() */
    setentry smalltable[PySet_MINSIZE];
    PyObject *weakreflist;      /* List of weak references */
} PySetObject;
```
{: file="cpython/Include/cpython/setobject.h" }

핵심은 `setentry`이다. `setentry`를 보면 `key`를 가지고 있고, 주석을 통해 `key`를 hash한다는 걸 알 수 있다. 즉, `PySet`은 `Hash Table`의 구조를 가지고 있다고 추측해 볼 수 있다. 

```c

/* set object implementation

   Written and maintained by Raymond D. Hettinger <python@rcn.com>
   Derived from Lib/sets.py and Objects/dictobject.c.

   The basic lookup function used by all operations.
   This is based on Algorithm D from Knuth Vol. 3, Sec. 6.4.

   ...
*/
```
{: file="cpython/Objects/setobject.c" }

`set`을 구현한 파일의 첫 주석이다. `set`은 `dict` 객체에서 파생되었다고 언급하고 있다. 이 주석을 통해 `set`이 `Hash Table` 구조를 사용한다는 것을 알 수 있다.

쉽게 말해, `PySet`은 `key`만 있는 `PyDict`이다.

---

## Go Map 분석

`Go`는 `map`이라는 자료구조를 제공한다. 이는 `{key:value}`로 매핑되는 `Hash table`이다. 이를 통해 `set`을 구현할 수 있을 것 같다. 그런데 `value`는 필요하지 않다. 따라서 value 자리에 `zero-value`를 넣어 마치 값이 없는 것과 같은 효과를 낼 수 있다. 

```go
// nil
map[T]interface{}{}
map[i] = nil

// struct
map[T]struct{}{}
map[i] = struct{}{}
```

대표적으로 `nil`과 `빈 struct`가 있다. 따라서 둘 중 어떤 값이 더 효과적일지 판단해야 한다.

### interface와 메모리 자원

```go
import (
	"fmt"
	"unsafe"
)

func main() {
	fmt.Println(unsafe.Sizeof(struct{}{}))  
    // 0

	var nilInterface interface{} = nil
	fmt.Println(unsafe.Sizeof(nilInterface))  
    // 16
}
```

`빈 struct`는 메모리 자체를 할당받지 않는다. 즉 메모리에 없는 값이다. 반면 `nil interface`는 16Byte를 할당받는다. 

그럼 빈 struct를 사용하면 메모리 할당이 적은가? 그건 또 아니다.

### Bucket의 작동원리

`map`이 어떻게 구현되어 있는지 살펴보자.

```go
// Map contains Type fields specific to maps.
type Map struct {
	Key  *Type // Key type
	Elem *Type // Val (elem) type

	Bucket *Type // internal struct type representing a hash bucket
}
```
{: file="go/src/cmd/compile/internal/types/type.go" }

`Map`은 `key:value` 쌍과 `Bucket`을 갖는다. 특징적인 점은 `Bucket`을 가진다는 것이다. 조금 더 깊게 들어가보자. 

```go
// A map is just a hash table. The data is arranged
// into an array of buckets. Each bucket contains up to
// 8 key/elem pairs. The low-order bits of the hash are
// used to select a bucket. Each bucket contains a few
// high-order bits of each hash to distinguish the entries
// within a single bucket.
//
// If more than 8 keys hash to a bucket, we chain on
// extra buckets.
//
// When the hashtable grows, we allocate a new array
// of buckets twice as big. Buckets are incrementally
// copied from the old bucket array to the new bucket array.

// mapextra holds fields that are not present on all maps.
type mapextra struct {
	// If both key and elem do not contain pointers and are inline, then we mark bucket
	// type as containing no pointers. This avoids scanning such maps.
	// However, bmap.overflow is a pointer. In order to keep overflow buckets
	// alive, we store pointers to all overflow buckets in hmap.extra.overflow and hmap.extra.oldoverflow.
	// overflow and oldoverflow are only used if key and elem do not contain pointers.
	// overflow contains overflow buckets for hmap.buckets.
	// oldoverflow contains overflow buckets for hmap.oldbuckets.
	// The indirection allows to store a pointer to the slice in hiter.
	overflow    *[]*bmap
	oldoverflow *[]*bmap

	// nextOverflow holds a pointer to a free overflow bucket.
	nextOverflow *bmap
}

```
{: file="go/src/runtime/map.go" }

앞으로 `key:value` 쌍을 `entry`라고 부르겠다. `bucket`은 8쌍의 entry를 담고 있다. 그리고 `map`은 bucket의 배열로 데이터를 저장한다. 만약 bucket 내의 entry가 유효한 포인터를 가지고 있지 않으면 `overflow bucket`을 생성해 값을 할당한다. `overflow bucket`에 할당된 entry는 더 이상 `GC(Garbage collector)`에 스캔되지 않는다. 

`빈 struct`는 포인터를 가질 수 없다. 따라서 `overflow bucket`으로 분류될 것이다. 따라서 GC에 의해 스캔되지 않는다. 하지만 `nil interface`는 포인터가 될 수 있기 때문에 계속해서 GC에 스캔된다. 이는 속도를 느리게 만든다.

반면 `overflow bucket`을 계속해서 생성하면서 지속적인 메모리 할당이 발생한다는 단점이 있다. 이를 해결하기 위해서는 `map`을 초기화할 때 `capacity`를 크게 잡으면 된다. `map`은 메모리 공간이 부족할 때 2배씩 늘려가며 메모리를 확보한다. 따라서 처음부터 넉넉하게 메모리를 할당해두면 bucket을 생성하는 빈도가 줄어든다. 

### 결론

결론적으로 `빈 struct`를 사용하는 것이 성능(속도/ 메모리) 면에서 더 뛰어나다. 만약 `map`의 `capacity`까지 넉넉하게 초기화해주면 더욱 좋은 성능을 보인다.

참고: [memory-allocation-and-performance-in-golang-maps](https://levelup.gitconnected.com/memory-allocation-and-performance-in-golang-maps-b267b5ad9217)

---

## Go로 구현

`PySet`의 주요 아이디어만 빌려와 `GoSet`을 만들어보자. 

```go
map[T]struct{}
```

`Go`에서는 `map`을 통해 `Hash Table`을 만들 수 있다. key만 가지는 map이 간단하게 구현되었다. 

간단한 코드로 `GoSet`이 잘 작동하는지 확인해보자. 

```go
import (
	"fmt"
)

func main() {
	data := []int{7, 3, 3, 5, 6, 1, 5}
    
	set := make(map[int]struct{})
	// add
	for _, n := range data {
		set[n] = struct{}{}
	}

	// print
	for key, _ := range set {
		fmt.Printf("%d ", key)
	}
	// 3 5 6 1 7 
}
```

예상대로 값이 출력되었다. 주의할 점은 `map`의 `key`는 순서를 유지하지 않는다. 이 부분은 `PySet`, `PyDict`도 동일하다.

### type으로 구현

추상화를 통해 `set`을 일반화 시켜보자. `Generic`을 사용해 다양한 타입의 데이터를 담을 수 있도록 작성했다. 

> Generic은 Go v1.18에 처음 추가된 기능이다. 따라서 1.17이하의 버전에서는 정확히 Key의 데이터 타입을 명시한 후 사용해야 한다.
{: .prompt-tip }

```go
// Set 구현
type Set[T comparable] struct {
	table map[T]struct{}
}

// 빈 Set 초기화
func NewSet[T comparable]() *Set[T] {
	emptySet := make(map[T]struct{})
	return &Set[T]{emptySet}
}

// Set에 값 추가
func (s *Set[T]) Add(key T) {
	s.table[key] = struct{}{}
}

// Set에 값이 있는지 확인
func (s *Set[T]) Has(key T) bool {
	_, ok := s.table[key]
	return ok
}

// Set에서 값 삭제
func (s *Set[T]) Pop(key T) bool {
	if s.Has(key) {
		delete(s.table, key)
		return true
	}
	return false
}

// Set에 모든 값 출력
func (s *Set[T]) PrintAll() {
	for key, _ := range s.table {
		fmt.Printf("%d ", key)
	}
	fmt.Print("\n")
}
```

이제 `Set`을 사용해보자. 

```go
package main

import "fmt"

func main() {
	data := []int{3, 5, 5, 6, 7, 7}
	set := NewSet[int]()

	for _, n := range data {
		set.Add(n)
	}
	set.PrintAll() // 3 5 6 7 
	fmt.Println(set.Has(5)) // true

	ok := set.Pop(7)
	if ok {
		set.PrintAll() // 5 6 3 
	}
}
```

예상한 대로 잘 작동하는 것을 볼 수 있다. 

---

## GoSet이 slice보다 빠를까?

중복을 제거할 때는 `Set`이 분명한 장점을 갖는다. 하지만 `Set`이 정말 탐색에서도 빠를까?

코드의 흐름은 다음과 같다. 

1. 9999999개의 정수를 각각 `slice`와 `Set`에 추가한다. 
1. 이때 추가되는 값은 100 이하의 랜덤한 정수이다. 
1. 마지막에 `slice`와 `Set`에 101을 각각 추가한다. (slice의 마지막에 101이 위치한다.)
1. `slice`와 `Set`에서 101을 탐색한다. 

```go
import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	lenData := 9999999
	maxRandInt := 100
	target := maxRandInt + 1

	testSlice(lenData, maxRandInt, target)
	testSet(lenData, maxRandInt, target)
}

func testSlice(lenData, maxRandInt, target) {
	// Slice 초기화 + 탐색 (최악의 경우)
	start := time.Now()

	data := make([]int, 0) // Set과 같은 조건에서 시작
	for i := 0; i < lenData; i++ {
		data = append(data, rand.Intn(maxRandInt))
	}
	data = append(data, target)

	for _, n := range data {
		if n == target {
			break
		}
	}
	duration := time.Since(start)
	fmt.Println("Slice으로 탐색", duration)
}

func testSet(lenData, maxRandInt, target) {
	// Set 초기화 + 탐색
	start := time.Now()

	set := NewSet[int]()
	for i := 0; i < lenData; i++ {
		set.Add(rand.Intn(maxRandInt))
	}
	set.Add(target)
	set.Has(target)
	duration := time.Since(start)
	fmt.Println("Set으로 탐색", duration)
}


// Slice으로 탐색 1.75740032s
// Set으로 탐색 1.343899904s
```

예상대로 `Set`이 빠르다. 하지만 `slice`의 크기가 작다면 어떨까? 

`lenData`를 99999로 줄이고 실행해봤다. 

```go
lenData := 99999
// Slice으로 탐색 9.400064ms
// Set으로 탐색 37.700096ms
```

`Set`이 훨씬 유리한 조건이었음에도 `slice`가 더 빠르다. 정확히는 `map`의 초기화가 느린 것이다.

같은 조건에서 초기화 시간을 제외하고 순수 탐색에 사용한 시간을 측정해 봤다. 

```go
lenData := 99999
// Slice으로 탐색 99.84µs
// Set으로 탐색 0s
```

`Set`이 더 빠르다. 따라서 탐색할 상황이 많다면 `Set`이 유리하다. 하지만 탐색이 많이 일어나지 않고, 데이터가 충분히 크지 않다면 오히려 무식하게 `slice`로 탐색하는 게 더 빠를 수 있다.
