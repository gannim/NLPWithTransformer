""" 2.1 데이터 처리 """

""" 2.1.1 데이터셋 (page 52~) """
# 미리 설치해야함! 참고 : https://github.com/rickiepark/nlp-with-transformers/blob/main/install.py
# transformers, datasets, accelerate, sentencepiece, umap-learn

''' init datasets '''
from datasets import load_dataset
from datasets import list_datasets

all_dataset = list_datasets()
print(f"dataset length : {len(all_dataset)}")
print(f"dataset[:10]\n{all_dataset[:10]}\n")

emotions = load_dataset("emotion")
train_ds = emotions['train']
print(f"emotions 객체\n{emotions}")
print(f"train_ds 객체\n{train_ds}")
print(f"train_ds 내 개별 샘플 참조\n{train_ds[0]}")
print(f"train_ds 내 colum 이름\n{train_ds.column_names}")
print(f"train_ds 내 features 속성\n{train_ds.features}")
print(f"train_ds 를 slice 연산자 이용해 몇개 행을 선택\n{train_ds[:5]}")
print(f"train_ds 이름으로 특정 열을 지정\n{train_ds['text'][:5]}")

''' 로컬에서 데이터셋을 로딩하는 스크립트 몇가지 '''
# load_dataset("csv", data_files="file.csv")
# load_dataset("text", data_files="file.txt")
# load_dataset("json", data_files="file.jsonl")
# load_dataset("csv", data_files="file.csv", sep=";", names=["text", "label"])
# load_dataset("csv", data_files="https://download.url.com/path/filename", sep=";", names=["text", "label"])

"""
2.1.2 dataframe (page 58~)
2.1.3 frequncy of Classes
2.1.4 length of tweet
"""

''' 데이터프레임 사용법 '''
emotions.set_format(type="pandas")
df = emotions["train"][:]
# 열 제목 출력후, 몇개의 행이 출력됨
print(df.head())

# 라벨을 int2str 로 라벨 이름에 해당하는 새로운 열을 데이터 프레임에 추가
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)
df["label_name"] = df ["label"].apply(label_int2str)
print("# add label str")
print(df.head())

''' 2.1.3 클래스 분포 확인 (page59-60) '''
import matplotlib.pyplot as plt
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("frequency oof classes")
plt.show()

# 실행해서 보면 데이터 불균형이 심한걸 알수있음
# oversampling, undersampling 을 통해 데이터 균형을 맞춤
# 샘플링 기법은 https://oreil.ly/5XBhb 참고
# 단, Train/Test set 분할전 샘플링 하지말 것, Train set 에만 적용

''' 2.1.4 트윗 길이 확인 (page60) '''
df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False,
           showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

# 그래프 확인해보면 감정 트윗의 길이는 15개 단어 정도
# 가장 긴 트윗도 DistilBert의 최대 문맥 크기보다 작음
# transformers에 적합한 포맷으로 변환하는 방법
emotions.reset_format()

""" 2.2 텍스트에서 토큰으로 """
import pandas as pd

''' 2.2.1 문자 토큰화 (page 63)'''
# 가장 간단한 토큰화 방법. 각 문자를 개별로 모델에 주입
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

# 수치화 (numericalization) 각 토큰을 고유한 정수로 인코딩 하는 것
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)

# vocabulary에 있는 각 문자를 고유한 정수로 바꾸는 mapping dictionary 만들기
input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)

# input_ids를 one-hot vector의 2D tensor로 변환
# one-hot vector는 ordinal, nominal 범주의 데이터를 인코딩하기 위해 자주 사용
# ex) 트랜스포머 TV 시리즈 캐릭터 이름을 인코딩 한다면 아래와 같이 작성 가능
categorical_df = pd.DataFrame({"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]})
print(categorical_df)

# 이런 형태의 코드는, 이름 사이에 가상의 순서가 생김, Neural Network는 이런 종류의 관계를 학습하는 능력이 뛰어남
# one-hot vector 형태로 변환
oh_vec = pd.get_dummies(categorical_df["Name"])
print(oh_vec)

# pytorch로 input_ids를 텐서로 바꾸고 One_hot() 함수를 사용해 원-핫 인코딩 생성
import torch
import torch.nn.functional as F
input_ids = torch.tensor(input_ids)
oh_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
# 38개 입력 토큰에 20차원의 one-hot vector 생성
# one_hot() 함수에 num_classes 매개변수 지정 중요, vocab보다 작아지는 문제가 생김
# num_classes 지정 안하면 입력 텐서에서 가장 큰 정수에 1을 더한 값을 클래스 개수로 사용
# ++ tensorflow 에는 tf.one-hot() 사용, depth 가 num_classes 와 동일 역할
print(oh_encodings.shape)

print(f"token: {tokenized_text[0]}")
print(f"tensor index: {input_ids[0]}")
print(f"one-hot encoding: {oh_encodings[0]}")

# 위와 같은 방식은 철자오류, 희귀한 단어 처리시 유용하나, 단어와 같은 언어 구조를 데이터에서 학습해야한다는 단점
# 학습 비용이 크기때문에 문자 수준의 토큰화는 거의 사용 않고, 단어 토큰화(word tokenization)를 사용

''' 2.2.2 단어 토큰화 (page 65)'''

# 단어를 사용하면 모델이 단어를 학습하는 단계 생략하게 되어 훈련 복잡도가 감소함
# 단어 토크나이저 1. 공백을 사용해 텍스트를 토큰화

tokenized_text = text.split() # line 9
print(tokenized_text)

# 이후 과정 문자 토큰화와 동일
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)
input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)
input_ids = torch.tensor(input_ids)
oh_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(oh_encodings.shape)

print(f"token: {tokenized_text[0]}")
print(f"tensor index: {input_ids[0]}")
print(f"one-hot encoding: {oh_encodings[0]}")

# 구두점이 고려되지 않아 하나의 토큰으로 처리되는 단점
# 곡용, 활용형, 철자 오류가 모두 포함되어, word vocab이 수백만개 까지 늘어나는 문제가 있음
# ++ 곡용(declination) 굴절어에서 문법 기능에 따라 단어 형태가 변하는것
# 어간(stem)으로 정규화 하는 어간추출(stemming), 표제어 추출(lemmatization) 적용 가능
# ++ great, greater, greatest -> great 로 변경

# word vocab이 커지면 neural network parameta가 많이 필요해저 문제가 발생 (66p)
# word vocab의 크기를 제한하는 일반적인 방법은, 빈도수 낮은 단어를 무시하는 것
# 말뭉치에서 자주 등장하는 10만개 단어만 사용, 그외에는 unk 토큰으로 매핑
# unk로 매핑하면 정보유실이 큼, 절충안으로 부분 단어 토큰화 방식이 나옴

''' 2.2.3 부분 단어 토큰화 (page 66) '''
# 드물게 나오는 단어를 더 작은 단위로 나누는 방식. 복잡한 단어나 철자 오류를 처리하기 용이함
# 이 토큰화 방식은, 통계 규칙과 알고리즘을 함께 사용해 사전 훈련 말뭉치에서 학습함

### WordPiece ###
# NLP에서 널리 사용되는 방식 중 하나, BERT와 DistilBERT의 토크나이저로 사용됨
# transformers package에서 AutoTokenizer 클래스를 제공
# from_pretrained() 메서드를 허브의 모델ID나 로컬 파일 경로와 함께 호출
## AutoTokenizer = 체크포인트 이름을 사용해, 모델의 설정, 가중치, vocab을 자동으로 추출하는 auto class임
## 이 클래스를 사용하면 모델 간의 빠른 전환이 가능ㅎ하지만, 특정 클래스를 수동으로 로드 가능

from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
encoded_text =tokenizer(text)
print(encoded_text)
"""
>> {'input_ids': [101, 19204, 6026, 3793, 2003, 1037, 4563, 4708, 1997, 17953, 2361, 1012, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""
# 문자 토큰화 처럼 단어가 input_ids필드에 있는 고유한 정수에 매핑됨
# attention_mask 필드는 다음 절에서 소개
# input_ids가 있으므로 토크나이저의 convert_ids_to_tokens() 메서드 사용, 이걸 다시 토큰으로 변환
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)
"""
>> ['[CLS]', 'token', '##izing', 'text', 'is', 'a', 'core', 'task', 'of', 'nl', '##p', '.', '[SEP]']
"""
print(tokenizer.convert_tokens_to_string(tokens))
"""
>> [CLS] tokenizing text is a core task of nlp. [SEP]
"""
# 1. [CLS], [SEP] 특수 토큰 추가 ~ 시퀀스의 시작과 끝을 알림
# 2. 토큰이 모두 소문자로 변홤됨
# 3. tokenizing, NLP가 2개의 단어로 쪼개짐 ~ 자주 쓰는 단어가 아니어서, ##은 문자열 앞이 공백이 아님을 뜻함

# AutoTokenizer 에서 제공하는 속성 몇가지
print(tokenizer.vocab_size)
print(tokenizer.model_max_length)
print(tokenizer.model_input_names) # 모델이 forward pass에서 기대하는 field name

''' 2.2.4 전체 데이터셋 토큰화 하기 '''
def tokenize(batch):
    # tokenizer를 샘플 배치에 적용하는 함수
    # padding=True : 배치에 있는 가장 긴 샘플 크기에 맞춰 샘플을 0으로 패딩
    # truncation=True : 모델의 최대 문맥 크기에 맞춰 샘플을 잘라냄
    return tokenizer(batch["text"], padding=True, truncation=True)

# corpus를 tokenize하기 위해 datasetDict 객체의 map() 함수 사용
from datasets import load_dataset
emotions = load_dataset("emotion")
print(tokenize(emotions["train"][:2]))
"""
>> {'input_ids': [[101, 1045, 2134, 2102, 2514, 26608, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[101, 1045, 2064, 2175, 2013, 3110, 2061, 20625, 2000, 2061, 9636, 17772, 2074, 2013, 2108, 2105, 2619, 2040, 
14977, 1998, 2003, 8300, 102]], 
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
"""
# input_ids 첫번째 원소가 두번째 보다 길이가 짧으므로 0이 추가됨
# [PAD] : 0, [UNK] : 100, [CLS] : 101, [SEP] : 102, [MASK] : 103
# attention_mask : 패딩 토큰때문에 모델이 혼동하지 않게 하려는 조치
# page 70. 그림2-3 확인

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
# map() 메서드 : 코퍼스 내 모든 샘플에게 개별적으로 작용
# batched=True 트윗을 배치로 인코딩
# batch_size=None 전체 데이터 셋이 하나의 배치로 tokenize() 함수 적용
print(emotions_encoded["train"].column_names)

# 배치에 있는 텐서에 동적으로 패딩 처리를 취하는 data collator 살펴보기
# 전역적인 패딩은 전체 코퍼스에서 feature matrix 추출할때 도움이 됨
"""
2.3 텍스트 분류 모델 훈련하기 (page 71)
"""
import pandas as pd

# 그림2-4 (page 72)
# 시퀀스 분류에 사용하는 encoder 기반 transformer architecture
# 1. text를 tokenize 해 token encoding이라는 one-hot vector로 나타냄
#    tokenizer word vocab 크기가 token encoding 차원을 결정함, 보통 2만~2백만개의 고유 토큰으로 구성
#    ++ pytorch에서는 one-hot vector 단계를 건너뜀, matrix에서 token id에 해당하는 열을 가지고 오는 형태로 대체함
#       3장 nn.Embedding 에서 확인해보는거로
# 2. token encoding 을 저차원 공간의 벡터인 token embedding으로 변환
# 3. token embedding을 encoder block layer에 통과 시켜 각 입력 token에 대한 hidden state를 만듦
#    각 hidden state는 language modeling의 pretrain objective(목표)를 달성하기 위해 masking된 입력 token을 predict 하는 layer로 전달
# 4. classification task 에서는 이 LM layer를 classification layer로 변경함

# 이런 데이터 셋에서 이런 모델 훈련하는 방법 2가지
# 1. feature extraction
#    사전 훈련 모델을 수정하지 않고, hidden state를 feature로 사용해 분류 모델을 훈련
# 2. fine tuning
#    사전훈련된 모델의 파라미터도 업데이트하기 위해 전체 모델을 end to end로 훈련

''' 2.3.1 transformer를 feature extractor 로 사용하기 '''
# 그림 2-5 (page 73)
# 훈련하는 동안 body의 weight를 동결, hidden state를 classification model의 feature로 사용
# 작거나 얕은 모델을 빠르게 훈련한다는 장점이 있음
# 이때 훈련되는 모델로 neural network classification layer 거나 random forest같이 gradient에 의존적이지 않은 기법이 있음
# hidden state를 한번만 미리 계산하면 되어서, gpu 사용 불가능할때 편리함
### 사전 훈련된 모델 사용하기 ###
# transformers package의 AutoClass인 AutoModel 사용
# AutoModel
# 사전 훈련된 모델의 가중치를 로드하는 from_pretrained() 메서드가 있음
import torch
from transformers import AutoModel
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    # tokenizer를 샘플 배치에 적용하는 함수
    # padding=True : 배치에 있는 가장 긴 샘플 크기에 맞춰 샘플을 0으로 패딩
    # truncation=True : 모델의 최대 문맥 크기에 맞춰 샘플을 잘라냄
    return tokenizer(batch["text"], padding=True, truncation=True)

from datasets import load_dataset

emotions = load_dataset("emotion")
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded["train"].column_names)

# gpu사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
# token encoding을 embedding으로 변환한 다음 encoder stack에 통과시켜서 hidden state를 반환

### 마지막 hidden state 추출 ###
text = "this is a test"
# return_tensors="pt" : token을 pytorch tensor로 변환, 지정하지 않으면 python list 전달
# tf 로 쓰면 tensorflow의 tensor 로 전달
inputs = tokenizer(text, return_tensors="pt")
# [batch_size, n_tokens]
print(f"input tensor size :  {inputs['input_ids'].size()}")

# 모델이 있는 장치로 옮기고 입력으로 전달
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    # gradient 자동 계산을 비활성화 하기위해 torch.no_grad() context manager를 사용
    # 계산에 필요한 메모리 양이 주로 prediction 할때 유리
    outputs = model(**inputs)
# 출력은 Python 의 namedtuple과 비슷한 형태로
# 모델설정에 따라 hidden state, loss, attention 같은 여러 객체를 포함함
print(outputs)
# hidden state size 확인, [batch_size, n_tokens, hidden_dim]
# 6개의 입력토큰마다 768차원의 벡터가 반환
# 분류작업에서는 보통 [CLS] token에 연관된 hidden state를 input feature로 사용
print(outputs.last_hidden_state.size())

# 전체 데이터 셋에서 같은 작업을 수행하고 hidden_state 열을 만들어 이런 vector를 모두 저장해보자
def extract_hidden_states(batch):
    # input을 gpu로 옮김
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        # 마지막 hidden_state 추출
        last_hidden_state = model(**inputs).last_hidden_state
    # [CLS] 토큰에 대한 벡터 반환
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

# 위 예시와 함수가 다른 점은 cpu로 가져와 numpy로 바꾸는거임, map에서 배치 입력을 사용하려면 python이나 numpy 객체를 반환하는 처리 함수 필요
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# 이 모델은 텐서가 입력되길 기대하므로 input_ids, attention_mask열을 "torch" 포멧으로 바꿈
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
# 모든 분할에 대해 hidden state를 한번에 추출
# batch_size=None으로 지정하지 않았기 때문에 batch_size=1000이 사용 (기본값)
print(emotions_hidden["train"].column_names) # extract_hidden_states함수 적용이후 hidden_state열이 데이터 셋에 추가

### 특성 행렬 만들기 ###
import numpy as np
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
Y_train = np.array(emotions_hidden["train"]["label"])
Y_valid = np.array(emotions_hidden["validation"]["label"])

print(X_train.shape, X_valid.shape)
# hidden state를 입력 특성으로 사용, label을 target으로 사용

### train set 시각화 하기 ###
# umap 알고리즘 사용, 이 벡터를 2D로 투영
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

# 특성 스케일을 [0,1] 범위로 조정
X_scaled = MinMaxScaler().fit_transform(X_train)
# UMAP 객체를 생성, 훈련함
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
# 2D 임베딩의 데이터 프레임 생성
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = Y_train
# 출력 결과는 훈련 샘플과 개수가 동일한 배열 , feature 는 2개로 나옴
df_emb.head()

# 각 범주에 대한 샘플 밀도를 개별로 그리기
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names

for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                   gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])

plt.tight_layout()
plt.show()

# sadness, anger, fear 같은 부정적인 감정은 조금식 다르게 분포, 차지하는 영역은 비슷
# joy, love은 부정적 감정과 잘 분리되고, 비슷한 영역을 차지
# surprise는 영역 전체에 골고루 퍼짐, 분리되길 기대했지만 실패
# 모델은 감정 차이를 구분하도록 훈련되지 않고 텍스트에서 마스킹된 단어를 추측해 암묵적으로 감정을 학습했을 뿐
# 데이터 셋의 특성을 간파했으니, 특성을 사용해 모델을 훈련해보자

### 간단한 분류 모델 훈련하기 ###
# hidden state가 감정별로 조금씩 차이남, 일부 감정에서는 명확한 경계가 없음
# 이 hidden state를 사용해 logistic regression 모델을 훈련
from sklearn.linear_model import LogisticRegression
# 수렴을 보장하기위해 max_iter를 증가시킴
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, Y_train)
lr_clf.score(X_valid, Y_valid) # 0.633
# 정확도는 모델이 무작위로 예측한것 보다 더 높은것 같음
# 불균형 다중 클래스 데이터 셋을 다루고 있어 실제 정확도는 더 좋음
# 이 모델이 얼마나 좋은지 단순한 기준 모델과 비교해보자

from sklearn.dummy import DummyClassifier
# 감정이 6개 있으므로, 무작위 예측시 정확도 16.5%, strategy="uniform" 지정해서 확인해볼수 있음
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, Y_train)
dummy_clf.score(X_valid, Y_valid) # 0.352

# 간단한 기준 모델 보다 훨씬 뛰어남
# 자세한 조사를 위해 confusion matrix를 보자
# 진짜 레이블과 예측 레이블의 관계를 보여줌

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, Y_valid, labels)

# anger와 fear는 sadness와 많이 혼동됨
# 임베딩을 시각화 했을때와 일치
# love, suprise는 joy로 많이 혼동됨
# 분류 성능을 높이기위한 fine-tuning은 다음절에서 확인


"""
2.3.2 트랜스포머 미세튜닝하기
"""
import pandas as pd
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

device = "cpu"
model_ckpt = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions = load_dataset("emotion")
labels = emotions["train"].features["label"].names

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# x_train = np.array(emotions_hidden["train"]["hidden_state"])
# x_valid = np.array(emotions_hidden["validation"]["hidden_state"])
# y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_encoded["validation"]["label"])

## 여기서부터 시작
# transformer를 end to end로 fine-tuning하는 조건
# 모델 전체를 훈련함, 이를 위해 분류 헤드는 미분이 가능해야함 -> neural network로 작업을 수행
# 분류 모델에 입력으로 사용하는 hidden state를 훈련하면 분류 작업에 적합하지 않은 데이터를 다룬다는 문제를 회피할수 있음
# 초기 hidden state는 훈련하는 동안 모델 손실이 감소하도록 수정되고, 성능이 높아짐
# transformers package 내 trainer api를 사용해 훈련 루프를 구현

### 사전 훈련된 모델 로드하기 ###

from transformers import AutoModelForSequenceClassification

# AutoModel과 달리, AutoModelForSequenceClassification 모델은
# 사전 훈련된 모델 출력 위에 base-model과 함께 쉽게 훈련할수 있는 분류 헤드가 있음

num_labels = 6  # 모델이 예측할 label 개수, 분류 헤드의 출력 크기
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

# 모델 일부가 랜덤하게 초기화 된다는 경고가 뜨지만, 상관없음. 분류 헤드가 아직 훈련되지 않아서 정상임

### 성공 지표 정의하기 ###
# 모델 성능 평가시 사용할 측정 지표를 정의

# Trainer에 사용할 compute_metrics()함수를 정의
# 이 함수는 EvalPrediction객체를 입력받아 측정 지표 이름과 값을 매핑한 딕셔너리를 반환
# f1-score와 accuracy를 계산

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


### 모델 훈련하기 ###

# Trainer 클래스 정의 전, 2가지 처리가 필요함
# 1. 허깅페이스 로그인. fine-tuning한 모델을 허브 계정에 저장, 커뮤니티에 공유
# 2. 훈련을 위한 모든 하이퍼 파라메터 정의

# ipython ide 에서 해야할듯
# from huggingface_hub import notebook_login
# notebook_login()
# 터미널에서는 huggingface-cli login 을 통해 access-token 입력
"""
huggingface-cli login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token: 
Add token as git credential? (Y/n) y
Token is valid.
Your token has been saved in your configured git credential helpers (osxkeychain).
Your token has been saved to /Users/seoga-eun/.cache/huggingface/token
Login successful
"""
# access-token 은 write 권한 이어야함
# git-lfs 설치 필요

from transformers import Trainer, TrainingArguments

# 훈련 파라메터 정의를 위해 TrainingArguments를 사용
# 이 클래스는 많은 정보를 저장, 훈련과 평가를 상세하게 제어
# 중요한 매개변수는 훈련과정에서 생성된 부산물이 저장될 output_dir임

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=True,
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  log_level="error")

# 이런 설정으로 Trainer 객체 생성, 모델을 fine-tuning 함
from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()

"""
{'eval_loss': 0.21856969594955444, 'eval_accuracy': 0.923, 'eval_f1': 0.9228438678785285, 
'eval_runtime': 50.9805, 'eval_samples_per_second': 39.231, 'eval_steps_per_second': 0.628, 'epoch': 2.0}
100%|██████████| 500/500 [56:29<00:00,  6.78s/it]
{'train_runtime': 3389.0922, 'train_samples_per_second': 9.442, 'train_steps_per_second': 0.148, 
'train_loss': 0.5299908294677734, 'epoch': 2.0}
{'test_loss': 0.21856969594955444, 'test_accuracy': 0.923, 'test_f1': 0.9228438678785285, 
'test_runtime': 52.8113, 'test_samples_per_second': 37.871, 'test_steps_per_second': 0.606}
"""

# 로그를 보면 검증셋에서 모델의 f1-score가 92%임. feature base 보다 많이 향상됨
# confusion matrix 로 좀더 자세히 보자
preds_output = trainer.predict(emotions_encoded["validation"])
# PredictionOutput 객체를 반환하는 함수, predictions, label_ids 배열과, Trainer 클래스에 전달한 측정 지표의 값도 담고 있음
print(preds_output.metrics)
# 각 클래스에 대한 예측 데이터도 있음
# 가장 큰 값이 나오도록 예측을 디코딩함, 그럼 예측 레이블이 반환되며, 이 결과값은 앞에 feature 기반 학습할때 sklearn 의 결과와 동일 포멧
y_preds = np.argmax(preds_output.predictions, axis=1)
# 이 예측을 사용해 confusion matrix 생성
plot_confusion_matrix(y_preds, y_valid, labels)
# 0에 가까운 형태로 매우 이상적임
# love는 여전히 joy와 혼동되지만 자연스럽
# suprise도 joy, fear와 혼동이 있음
# 전반적으로 우수해 보임

### 오류 분석 ###
# 오류 유형을 자세히 분석해보자
# 손실 기준으로 검증 샘플을 정렬하는 방법
#  정방향 패스의 결과와 레이블을 사용하면 손실은 자동으로 계산 가능

from torch.nn.functional import cross_entropy


def forward_pass_with_label(batch):
    # 손실과 예측 레이블을 반환하는 함수
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    # 모든 입력 텐서를 모델과 같은 장치로 이동

    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction="none")
    return {"loss": loss.cpu().numpy(),
            "predicted_label": pred_label.cpu().numpy()}
    # 다른 데이터셋 열과 호환되도록 출력을 CPU로


# 다시한번 map()으로 위 함수를 적용해 모든 샘플의 loss를 구함
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)

# 텍스트, 손실, 예측 레이블과 진짜 레이블로 DataFrame 생성
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"]
                              .apply(label_int2str))

# emotions_encoded를 손실 기준으로 오름차순이나 내림차순으로 정렬
# 이 작업을 통해 아래와 같은 사항을 감지함
# 1. 잘못된 레이블 감지
# 2. 데이터셋의 특이사항 감지
#    모델의 나쁜 예측을 들여다보면, 입력에 포함된 특수문자, 문자열이 모델 예측에 큰 영향을 미치는것을 발견할수 있음

df_test.sort_values("loss", ascending=False).head(10)
# 데이터를 보면, 일부 레이블 예측이 틀린것도 보이고, 애매한것도 있어보임
# joy는 여러번 잘못 레이블링 된것도 보임
# 위 정보를 바탕으로 데이터를 정제하거나, 추가거나, 더 큰 모델을 사용하면 성능 향상에 도움이 됨

# 가장 낮은 손실을 내는 예측
df_test.sort_values("loss", ascending=True).head(10)
# sadness 클래스 예측시 확신이 강함

### 모델 저장 및 공유 ###
trainer.push_to_hub(commit_message="Training completed!")

# fine-tuning 모델을 사용해 새로운 트윗 예측
from transformers import pipeline

model_id = ""
classifier = pipeline("text-classification", model=model_id)
custom_tweet = "I saw a movie today and it was really good"
preds = classifier(custom_tweet, return_all_scores=True)
# 각 클래스별 확률을 막대 그래프로 나타내기
preds_df = pd.DataFrame(preds[0])
plt.bar(labels, 100 * preds_df["score"], color='C0')
plt.title(f"{custom_tweet}")
plt.ylabel("class probability (%)")
plt.show()

""" 
2.4 결론 

모델을 업로드하면 http 요청을 받는 추론 엔드포인트가 자동으로 생성됨! 
 * https://huggingface.co/docs/api-inference/index 참고 바람!

더 빠른 예측을 위한 기법은 8장에서! 

트랜스포머로 다양한 작업을 학습!

다중언어 지원함! (4장)
"""
