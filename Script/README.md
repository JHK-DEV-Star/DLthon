# EDA

- **주어진 task**: text classification
  - 전체 문장 데이터 길이 분포 (raw data)

    <img src="/img/img01.png" width="700px" height="450px"></img>

  - Class 별 문장 데이터 길이 분포 (raw data)

    <img src="/img/img02.png" width="700px" height="450px"></img>


    | 데이터 타입 | 문장 최소 길이 (Min) | 문장 평균 길이 (Mean) | 문장 최대 길이 (Max) |
    |---|---|---|---|
    | **Train** |9|46.72|223|
 
    <br>
 
    | 문장 분류 | 문장 최소 길이 (Min) | 문장 평균 길이 (Mean) | 문장 최대 길이 (Max) |
    |---|---|---|---|
    | **협박 대화** |12|62|223|
    | **갈취 대화** |9|55|185|
    | **직장 내 괴롭힘 대화** |9|59|213|
    | **기타 괴롭 대화** |9|53|222|


- 데이터 갯수:

    <img src="/img/img03.png" width="700px" height="700px"></img>

  | | 협박 대화 | 갈취 대화 | 직장 내 괴롭힘 대화 |기타 괴롭 대화|
    |---|---|---|---|---|
    | 갯수 |896|1,094|981|979|


# Data 특징

- **train data**: 멀티턴 대화. 줄바꿈 문자(\n) 로 화자를 구분하고 있음
   - **예시**: 친구 돈 있어?\n저요?\n웃어. 사람들이 오해해\n왜그러세요?\n내가 교통비가 없어서 그래 돈 조금만 줘\n저 돈 없어요\n찾아서 나오면?\n왜그러세요 죄송합니다\n이거라도 가져갈게\n돌려주세요!

<br>

- **test data**: 멀티턴 대화. 화자를 구분하지 않음(화자 구분이 가능한 문자 없음)
   - **예시**:야 박보영 오늘 학원 갔다왔어 아니 오늘 못갔어 왜 아니 친구랑 놀다가 시간을 못봤어 자랑이냐 깜빡했어 저녁은 뭐야 너는 저녁 없어 학원 다녀왔어야지 엄마 미안해 나 배고파 다음부터 꼭 가 알았어 나 손 씻고올게 깨끗하게 씻고와

# 합성 데이터 수집
- 내용: 일반 대화, 2211(set)
- 활용: gemini, GPT
- 한번에 20회 이하의 대화set을 반복 수집
- 슬랭, 욕등을 섞어서 생성
- 기존 데이터의 분포와(문장 길이, 화자 수, 대화 횟수) 유사하게 생성

# 데이터 전처리
- preprocess_sentence
   - 문장 단위 정규화
   - 소문자화, 공백 정리(양끝 공백제거, 특수문자 앞뒤 공백, 다중 공백 하나로), 괄호 속 텍스트 삭제, 줄바꿈 제거

- preprocessing
   - 중복/결측 제거(결측치가 있으면 행 삭제)
   - KC-BERT 기반 단어 삽입 aurgumentation
   - preprocess_sentence 적용
   - 문장 중복 제거


# 모델 설계

- 모델 선택

주어진 task는 문장을 전부 입력 받아서 특정 분류의 class로 반환해아 합니다. Transformer 모델은 NLP에서 문장의 길이에 의존하지 않고 좋은 결과를 반환하는 모델입니다. 따라서, 저희는 gradient vanishing, long term dependancy 문제에서 벗어나고자 transformer 모델을 선택하였습니다. Transformer 모델은 BERT와 같은 encoder 기반의 모델과 GPT와 같은 decoder 기반의 모델이 있습니다. 이 부분에 있어서 pretrained 되지 않은 모델의 비교에서 decoder 모델보다 encoder 모델이 더욱 뛰어난 성능을 보여줬음으로 decoder 모델을 패기 하고 encoder 모델을 선택하였습니다.

| 모델 타입 | f1 | 
|---|---|
| **Encoder Base** |0.98|
| **Decoder Base** |0.81|

이때, 저희는 중요한 맥락을 보면 좋겠다고 생각하였습니다. 다음과 같은 예시를 보면 일반적인 대화와 위협 대화 중 구분하기 힘든 대화들이 존재합니다.

| class | conversation | 
|---|---|
| **직장 내 괴롭힘 대화** |한주씨 주말에 뭐해?<br>아 대구 본가 내려갑니다<br>아그래? 잘됏다 우리집도 그쪽이거든 거창 알아?<br>아네 알아요<br>그럼 주말에 오가면서 나좀 태워서 가<br>네??<br>나도 집에 갈일있어서<br>아 차에 카시트랑 짐이랑 너무 많아서 자리가 없는데<br>좀 치우고 불편하게 가면되지 대신 이야기하면서 재밌게 가면되잖아?<br>아. 와이프랑 애들이 불편할것같은데<br>내가 더불편하지 가족들 사이에 낀거니까 그렇지?<br>아.네.<br>그날보게<br>|
| **일상대화** |교수님, 질문 있습니다!<br>네, 학생. 뭐죠?<br>이번 과제... 주제가... 너무... 어렵습니다.<br>어렵다고요? 뭐가 어렵죠?<br>너무... 포괄적입니다! 좀... 좁혀주시면 안 될까요?<br>하하. 원래 그런 의도였습니다. 알아서들 해보세요.<br>아...<br>이상. 다음.<br>...교수님... 밉다.<br>...|
| **일상대화** |야, 시험 개망함.<br>나도. 한 문제도 모르겠더라. ㅅㅂ.<br>교수님 왜 저러시냐. 미친 듯.<br>ㄹㅇ. 이거 재수강 각임?<br>아... 밥맛 떨어진다.<br>술이나 마시러 가자.<br>콜. 오늘은 마셔야겠다.<br>가자. 쏘주 ㄱ.<br>ㅇㅋ.<br>짠.|


이러한 위협 문장은 시각에 따라서 달라 질 수 있으나 대화 당사자 기준으로 볼 시 일상 대화가 될 수 있습니다.

- 모델 architecture

모델의 architecture는 이전에도 언급하였듯이 BERT의 encoder 모델을 사용하였습니다. Masked Multi-Head attention이 효율적이지 못한 측면도 있고 실험 결과적으로 encoder base 모델이 더 나은 성능을 보여주었습니다.

<img src="/img/img04.png" width="600px" height="800px"></img>


- Activation Funciton

ReLU의 경우 Dead Neuran 문제 때문에 일부 node 에서 계산이 되지 않는 문제가 발생할 수 있습니다. 따라서, 더 효율적인 activation function인 SiLU를 사용하였습니다. 이는 GELU와 비슷한 성능을 내면서 기존의 다른 activation funciton이 갖고 있는 여러 문제점들을 해결하였습니다.

<img src="/img/img05.png" width="600px" height="360px"></img>

- Positional Embedding

Positional embedding의 경우 absolute positional encoding의 sinous positinal encoding 대신 RoPE를 사용하여 단어의 상대적인 위치를 알 수 있게 하였습니다. RoPE의 경우 Q vector와 K vector를 회전시켜 그 위치를 기억함으로 같은 token이라도 위치가 다르면 회전값이 다르기 때문에 다른 값을 갖게 됩니다.

# 시도 방법

- 기본 적인 목표 달성을 위하여 BERT 모델을 pretarin 과정 없이 text classification에 맞춰서 모델을 설계

   - 결과:
     | model | f1 | 
     |---|---|
     |basic|0.48|

- 일상 대화에 대한 구분 능력이 부족한 것 같아 \<mask> token을 도입

   - 결과:
     basic과 성능에 크게 변화가 없어 측정 하지 않음

- Vocabulary size 증가

   - 결과:
     | model | f1 | 
     |---|---|
     |vocab_size_16384|0.67|
- 일상 대화 데이터량 증가

  - 결과:
    실험시간이 오래걸려 끝마치지 못함
    
- 전체 데이터량을 augmentation으로 50,000개 까지 증가

  - 결과:
    실험시간이 오래걸려 끝마치지 못함
    
- Auxiliary loss의 objective 수정. BERT와 같이 auxiliary loss를 \<mask> 된 문장을 입력 받아서 원래 문장을 맞추는 task로 변경

  - 결과:
     | model | f1 | 
     |---|---|
     |vocab_size_16384|0.68|

- Pre-Normalization 사용, GLU 구조 사용, 잘못된 RoPE 변경

   -결과:
    적은 학습량으로 이전과 같거나 비슷한 성능. 아직 hyperparameter 수정이 끝나지 않음
     | model | f1 | 
     |---|---|
     |model_revision|0.67|

# 실수

- mask를 구현할 때 for 문 밖으로 빼지 random하게 선택 된 마지막 index만 mask 처리 되고 있었음
- padding mask를 마지막 layer에 넣어줄 때 padding을 고려하지 않았음
- RoPE 구현할 때 실수가 너무 많았음
- 실수를 바로 잡고 모델을 돌려 본 결과 학습 데이터의 분포를 잘 학습하는 것을 확인하여 학습 데이터의 비율이 정 비율일 때 가장 효율이 좋다는 것을 알게 되었음

# 결론
- augmentation을 한 후 validation set을 랜덤하게 나누면서 정보가 유출되어 학습 시 충분히 학습하지 못하였거나 과하게 학습했을 가능성이 있습니다.

- Pre-Normalization, RoPE, SwiGLU 등의 구조를 encoder base 의 transformer에 적용했을 때 확실히 모델의 성능이 증가하였습니다.

- 중간 실수로 인한 실험 시간 지연으로 추가적인 hyperparameter tunning에 대한 실험이 필요합니다.
