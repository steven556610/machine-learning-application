電子郵件垃圾郵件分類專案專案目標本專案旨在透過對電子郵件進行垃圾郵件分類，作為學習自然語言處理 (NLP) 分析的實踐案例。專案的核心目標是建構一個能夠將電子郵件自動識別為「垃圾郵件」（Spam）或「非垃圾郵件」（Ham）的機器學習模型。專案架構與技術棧專案文件本專案的程式碼和相關檔案已上傳至(https://github.com/steven556610/machine-learning-application/tree/main/text_ml) 儲存庫。

環境建置專案環境主要使用 pip 進行套件管理，並已成功建立包含以下核心函式庫的環境檔案 (.yml)：numpy tensorflow torch transformers參考資源本專案參考了多個學術與實踐資源，以確保理論與實務的結合。

Jupyter Notebook 範例:(https://www.geeksforgeeks.org/nlp/detecting-spam-emails-using-tensorflow-in-python/)
dataset: (https://www.kaggle.com/datasets/zeeshanyounas001/email-spam-detection/data)

教科書:Eisenstein, J. (n.d.). Introduction to Natural Language Processing.
(https://karczmarczuk.users.greyc.fr/TEACH/TAL/Doc/Handbook%20Of%20Natural%20Language%20Processing,%20Second%20Edition%20Chapman%20&%20Hall%20Crc%20Machine%20Learning%20&%20Pattern%20Recognition%202010.pdf)(https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)

參考網頁:
(https://www.tempmail.us.com/en/parsing/deciphering-emails-with-python-and-natural-language-processing)Fetching and Analyzing Mail from Gmail Using PythonNLP Libraries in Python
(https://blog.csdn.net/yosemite1998/article/details/122306758)Natural Language Processing for Emails資料處理流程 (Data Processing Pipeline)

本專案的資料前置處理步驟，是確保模型能夠有效學習和預測的關鍵。
以下是主要處理流程：
1. 文字清理 (Text Cleaning)在模型訓練前，對原始電子郵件文本進行初步清理，以移除雜訊並保留關鍵資訊：移除標題: 移除文本開頭的 "Subject:" 字串。移除停用字 (Stopwords): 使用 nltk.corpus 中的 stopwords 列表，過濾掉如 "the"、"a" 等不具備分類語義的常用詞彙。
2. 編碼 (Encoding) 與分詞 (Tokenization)TensorFlow Tokenizer:Tokenizer 是一個核心工具，用於將文本中的每個單字映射到一個唯一的整數 ID，從而建立一個詞彙表。fit_on_texts(): 這是 Tokenizer 的學習階段。
其原理是迭代所有訓練文本，執行以下操作：分詞 (Tokenization): 預設使用空白符號將文本分割成詞元 (token)。
* 建立詞彙表: 收集所有不重複的單字，並為其分配唯一的整數 ID。此映射關係儲存在 tokenizer.word_index 屬性中。
* 單字頻率統計: 同時統計每個單字在語料庫中的出現頻率。
* texts_to_sequences(): 這是 Tokenizer 的轉換階段。它利用 fit_on_texts() 建立好的詞彙表，將新的文本轉換為整數序列。
3. 序列填充與截斷 (pad_sequences)目的: 由於深度學習模型（如 RNNs 和 LSTMs）的輸入張量維度必須固定，pad_sequences() 用於將所有長度不一的整數序列，轉換為一個統一長度的 2D NumPy 陣列。
機制:
* 填充 (Padding): 對於長度短於指定 maxlen (在本專案範例中為 100) 的序列，它會在序列末尾 (padding='post') 補上 0，直到達到 maxlen 長度。
* 截斷 (Truncating): 對於長度長於 maxlen 的序列，它會從序列末尾 (truncating='post') 移除多餘的部分，直到長度符合 maxlen。以下為核心程式碼範例：

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)

max_len = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
```
4. PyTorch 中的對應功能在 PyTorch 生態中，類似的文本處理功能通常由不同的函式或套件組合實現：分詞器: torchtext.data.utils.get_tokenizer 或使用 spaCy 等外部套件。建立詞彙表: 需要手動完成，通常是先使用分詞器處理文本，再用 collections.Counter 統計單字，並建立單字到整數的映射字典。填充序列: torch.nn.utils.rnn.pad_sequence 函式。

未來展望 (Future Work)本專案將持續演進，未來的研究方向包括：深入研究教科書內容: 仔細研讀 NLP 相關教科書，以深化理論基礎。探索不同模型架構: 嘗試使用不同於當前範例中的 LSTM 和 RNN，例如 Transformer 模型。研究不同編碼方式: 探索如 Word2Vec 等非固定向量編碼方法。提升外部資料集效能: 針對在外部測試集上表現不理想的問題，分析其原因（如過度擬合或擬合不足），並尋找改進策略。模型可解釋性 (Feature Explainability): 探索如何解釋模型做出的預測。部署與維護: 將模型部署為 API，並考慮使用 Docker 和 MLOps 概念來建立一個穩健的 ML 服務。