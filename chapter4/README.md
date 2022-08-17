## 4장 중간부터 학습하자! 사전 학습과 파인튜닝

사전 학습의 사전은 미리라는 의미를 지닌 사전이다.

파인튜닝은 이미 만들어져 있는 것을 조금 변형하는 것이다.

BERT 구조 뒤에 각자 만들고자 하는 모델의 성격에 따라 레이어를 추가해주면 끝이다.

BERT 이후에 나오는 트랜스포머 기반의 언어 모델은 거의 모두 사전 학습 후 파인튜닝하는 구조로 이뤄져 있다.

<img src="images/BERT_모델_전체_구조.png">

### BERT 모델의 입력 이해하기

BERT의 입력 데이터는 평뮨을 토큰화하는 것부터 시작한다.

BERT는 워드피스 포크나이저를 사용한다.

- 한 문장을 토큰화할 때 우선 더 이상 쪼갤 수 없는 유닛 단위로 쪼갠 후 인접하는 유닛들끼리 합쳐가면서 토큰을 만드는 알고리즘

각 토큰은 번호를 갖고 있다.

"There is my school and I love this place" 라는 문장에 대한 전체 구조는 다음 그림과 같다.

<img src="images/BERT_입력에_대한_전체_구조.png">

### 사전학습은
- MLM(Masked Language Model) 과 NSP(Next Sentence Prediction) 을 통해 학습한다.

<img src="images/MLM_학습_과정_요약.png">

<img src="images/NSP_학습_데이터_예시.png">

<img src="images/BERT_사전_학습_요약.png">

### 텍스트 분류 모델로 파인튜닝하기

- CoLA 데이터셋 사용: Corpus of Linguistic Acceptability
- transformers 라이브러리에 있는 BertForSequenceClassification 클래스 사용
- 추론 결과를 confusion matrix 로 표현 가능
- 평가 방법은 Matthews Correlation을 사용

### 질의응답 모델로 파인튜닝하기

- SQuAD 데이터셋 사용: Stanford Question Answering Dataset
- transformers 라이브러리에 있는 BertForQuestionAnswering 클래스 사용
- F1 스코어 확인

### GPT (Generative Pre-Training), 생성하는 사전 학습

- 위에서 설명한 BERT 는 Transformer 의 인코더 부분을 이용해서 만든 모델
- GPT 는 Transformer 의 디코더 부분을 이용해서 만든 모델

- 앞에 나온 단어를 이용해서 다음 단어를 맞춰나가는 방식으로 사전 학습을 진행

<img src="images/GPT를_이용한_언어_모델_학습_방법.png">

<img src="images/BERT_vs_GPT.png">

<img src="images/GPT의_Masked_Self-Attention_에서의_마스킹_방법.png">

### RoBERTa

Facebook 에서 2019년 7월에 발표된 논문

"RoBERTa: A Robustly Optimized BERT Pretraining Approach"

BERT를 최적으로 학습하는 것이 핵심
- BERT의 구조와 동일
- MLM 학습할 때 데이터를 처리하는 방법
- NSP 를 처리하는 방법
- 배치 사이즈 조절
- 토크나이저 변경 
등의 방법을 이용

<img src="images/RoBERTa의_Dynamic_Static_마스킹.png">

<img src="images/RoBERTa에서의_NSP_전략.png">

### ALBERT

2019년 7월에 Google Research 와 Toyota Technological Institute at Chicago 가 연구해서 발표한 언어 모델

"A Lite BERT for Self-supervised Learning of Language Representations"

BERT 의 모델 사이즈가 크다는 단점을 극복한 언어 모델

모델 사이즈를 줄이는 방법
- Factorized Embedding Parameterization
- Cross-layer Parameter Sharing

Factorized Embedding Parameterization
- BERT 에서는 임베딩 사이즈를 히든 사이즈와 같게 뒀다. (<BERT_입력에_대한_전체_구조> 참고)
- 임베딩 사이즈(E) 와 히든 사이즈 (H) 를 굳이 같은 값을 묶을 필요가 없다고 이야기하고 있다.

<img src="images/ALBERT와_BERT의_임베딩_파라미터_수_계산.png">

Cross-layer Parameter Sharing
- BERT 에서는 Self-Attention을 계산해서 H 차원의 결과값을 만들어내는 BertLayer 를 12번 반복한다.
- 이 때 파라미터를 공유하지 않으므로 같은 구조의 블록을 12개 만든다.
- ALBERT 에서는 이 구조의 블록을 1개 만들어서 재사용한다. -> 결과적으로 파라미터 공유

<img src="images/ALBERT와_BERT의_인코더_동작_구조.png">

또 하나 짚고 넘어가야 할 부분인 "SOP (Sentence Order Prediction)"

<img src="images/SOP_학습_데이터_예시.png">

### ELECTRA

2020년 5월에 Google Brain 과 스탠퍼드 대학교가 함께 연구해서 발표한 언어 모델

"Efficiently Learning an Encoder that Classifies Token Replacements Accurately"

학습 효율을 개선함으로써 사전 학습 시간을 현저히 줄임

ELECTRA 에서는 MLM 대신 RTD(Replaced Token Detection) 이라는 방법을 사용해서 학습

ELECTRA 를 학습할 때도 Generator 와 Discriminator 가 필요하다.

<img src="images/ELECTRA의_Generator와_Discriminator_역할.png">

즉, MLM 은 전체 입력 토큰의 15% 에 대해서만 학습 -> 이게 비효율적이라고 생각  
RTD 에서는 모든 토큰이 학습 대상이 됨 (fake or real) -> 하나의 입력 데이터로 더 많은 학습 가능

<img src="images/RTD를_통한_언어_모델_학습_효과.png">

### DistilBERT

지금까지 설명한 BERT 기반 모델의 크기도 굉장히 커졌다. -> 자연스럽게 연산량 증가 -> 지연 시간이 길어짐

가벼운 모델을 만들어보기 위한 경량화에 대한 연구도 활발하게 진행
- 양자화 (Quantization)
- 가지치기 (Pruning)
- 지식 증류 (Knowledge Distillation)

DistilBERT 는 BERT 를 지식 증류 기법으로 경량화한 모델

<img src="images/지식_증류_시_사용하는_목적_함수.png">

BERT 와 비교했을 때 파라미터 수도 2배 정도 적고, 속도는 2배 정도 빠르다.

<img src="images/BERT_RoBERTa_DistilBERT_학습_시간_비교.png">

### BigBird

BERT 의 연산량에 가장 큰 영향을 미치는 것이 어텐션이다.

이 어텐션의 연산량을 줄이기 위한 많은 연구가 있었다.

Sparse 어텐션에 대해 이야기 해 보고, BigBird 라고 하는 긴 시퀀스를 위한 트랜스포머에 대해서도 알아보자.

BigBird 에서의 어텐션은
- 글로벌 어텐션
- 로컬 어텐션
- 랜덤 어텐션
으로 구성된다.

<img src="images/BigBird의_어텐션과_셀프어텐션.png">

<img src="images/BigBird와_RoBERTa의_토큰_길이에_따른_지연_시간.png">

### 리포머 (Reformer)

셀프어텐션은 우수한 성능을 보여주지만 연산량이나 메모리 사용량 측면에서는 비효율적이다. 

2020년에 구글에서 이 문제를 해결하기 위해 발표한 논문

Reformer Transformer

<img src="images/기존_트랜스포머_기반의_모델이_갖는_문제점.png">

리포머에서는 위 3가지 문제점을 개선

- LSH (Locally Sensitive Hashing) 어텐션

<img src="images/기존_트랜스포머에서의_어텐션_계산.png">

<img src="images/LSH의_핵심_아이디어.png">

<img src="images/앵귤러_LSH.png">

<img src="images/LSH를_이용한_어텐션_계산하기.png">

<img src="images/청크_단위로_묶어서_어텐션_계산하기.png">

- Reversible 트랜스포머

많은 메모리가 필요한 문제를 해결한 방식

활성화 함수 값을 메모리에 저장하지 않고 계산해서 쓸 수 있는 RevNet을 응용해서 Reversible Transformer 제안함

<img src="images/ResNet과_RevNet의_구조.png">

### GLUE 데이터셋

General Language Understanding Evaluation

<img src="images/GLUE_데이터셋_정리.png">

