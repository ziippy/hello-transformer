## 4장 중간부터 학습하자! 사전 학습과 파인튜닝

사전 학습의 사전은 미리라는 의미를 지닌 사전이다.

파인튜닝은 이미 만들어져 있는 것을 조금 변형하는 것이다.

BERT 구조 뒤에 각자 만들고자 하는 모델의 성격에 따라 레이어를 추가해주면 끝이다.

BERT 이후에 나오는 트랜스포머 기반의 언어 모델은 거의 모두 사전 학습 후 파인튜닝하는 구조로 이뤄져 있다.

<img src="BERT_모델_전체_구조.png">

### BERT 모델의 입력 이해하기

BERT의 입력 데이터는 평뮨을 토큰화하는 것부터 시작한다.

BERT는 워드피스 포크나이저를 사용한다.

- 한 문장을 토큰화할 때 우선 더 이상 쪼갤 수 없는 유닛 단위로 쪼갠 후 인접하는 유닛들끼리 합쳐가면서 토큰을 만드는 알고리즘

각 토큰은 번호를 갖고 있다.

"There is my school and I love this place" 라는 문장에 대한 전체 구조는 다음 그림과 같다.

<img src="BERT_입력에_대한_전체_구조.png">

### 사전학습은
- MLM(Masked Language Model) 과 NSP(Next Sentence Prediction) 을 통해 학습한다.

<img src="MLM_학습_과정_요약.png">

<img src="NSP_학습_데이터_예시.png">

<img src="BERT_사전_학습_요약.png">

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


