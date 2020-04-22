# Wikipedia日本語版全文を学習したDistilBERTモデル

## はじめに
バンダイナムコ研究所技術開発本部のライです。今回は[Wikipedia日本語版](https://ja.wikipedia.org/)全文を学習したDistilBERTモデルを公開するとともに、PretrainとFinetuningの手法を紹介します。

## DistilBERTとは?
DistilBERTは[Huggingface](https://huggingface.co/) が NeurIPS 2019 に公開したモデルで、名前は「Distilated-BERT」の略となります。投稿された論文は[こちら](https://arxiv.org/abs/1910.01108)をご参考ください。

DistilBERTはBERTアーキテクチャをベースにした、小さくて、速くて、軽いTransformerモデルです。DistilBERTは、BERT-baseよりもパラメータが40%少なく、60%高速に動作し、GLUE Benchmarkで測定されたBERTの97%の性能を維持できると言われています。

DistilBERTは、教師と呼ばれる大きなモデルを生徒と呼ばれる小さなモデルに圧縮する技術である知識蒸留を用いて訓練されます。BERTを蒸留することで、元のBERTモデルと多くの類似点を持ちながら、より軽量で実行速度が速いTransformerモデルを得ることができます。

## 事前学習
[公式ガイダンス](https://github.com/huggingface/transformers/tree/master/examples/distillation)を参考にしながら6-layer, 768-hidden, 12-heads, 66M parametersのモデルを学習しました。Wikipediaのテキスト抽出は[Wikiextractor](https://github.com/attardi/wikiextractor)を使いました。

教師となるBERT-baseモデルは東北大学 乾・鈴木研究室によって作成・公開されたBERTモデルを使っていますので、Tokenizerもセットで`bert-base-japanese-whole-word-masking`を使いました。生徒モデルのベースは`distilbert-base-uncased`を使っています。

https://www.nlp.ecei.tohoku.ac.jp/news-release/3284/

https://github.com/cl-tohoku/bert-japanese

事前学習用のパラメータは以下の通りです。

```bash
python transformers/examples/distillation/train.py \
    --student_type distilbert \
    --student_config training_configs/distilbert-base-uncased.json \
    --teacher_type bert \
    --teacher_name bert-base-japanese-whole-word-masking \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path $MODEL_PATH \
    --data_file $BINARY_DATA_PATH \
    --token_counts $COUNTS_TOKEN_PATH \
```

## 利用方法

### 必要パッケージ

```
torch>=1.3.1
torchvision>=0.4.2
transformers>=2.5.0
tensorboard>=1.14.0
tensorboardX==1.8
scikit-learn>=0.21.0
mecab-python3
```
※transformersについて、AutoModelが利用可能な2.5.0と現時点(2020/04/14)最新版の2.8.0では動作確認しましたが、将来の更新で公式のソースコードが変更される可能性がありますので、ご了承ください。

### GitHub からダウンロード

[DistilBERT-base-jp.zip](https://github.com/BandaiNamcoResearchInc/DistilBERT-base-jp/releases) をダウンロードし、解凍してください。

※モデルファイルが100MBを超えたため、リポジトリをcloneする場合Git LFSが必要です。詳細は[こちら](https://git-lfs.github.com/)をご参考ください。

```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-japanese-whole-word-masking")
model = AutoModel.from_pretrained("LOCAL_PATH")                                     
```
LOCAL_PATHは上記ファイル解凍後のパスのことです。以下3ファイルが含まれることを確認してください：

- pytorch_model.bin
- config.json
- vocal.txt

### Transformers ライブラリーから読込み

```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-japanese-whole-word-masking")
model = AutoModel.from_pretrained("bandainamco-mirai/distilbert-base-japanese")     
```

Pytorch用モデルのみの公開となりますが、HuggingFaceのtransformersライブラリを用いれば Tensorflow 2.0 形式に変換することもできます。

### 自前のデータセットでFine-Tuning

Transformersライブラリではfine-tuningするサンプルがいくつありますが、全部英語のデータを対象としているため、日本語データを使うならソースコードを修正する必要があります。ここはLivedoorニュースコーパスの分類タスクを例として説明します。

※ここまでの手順ですでに pip で transformers をインストールしていますが、ここからの例では GitHub から clone してきたコードを修正して実行するので、競合を避けるため pip uninstall transformers を実行してください。

まず、[Transformers](https://github.com/huggingface/transformers) からcloneした`transformers/src/transformers/data/processors/glue.py`にタスクのprocessorを追加。

```python
class LivedoorProcessor(DataProcessor):
    """Processor for the original data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
```
※ここで学習データ(train)と検証データ(dev)は [テキスト\tラベル] 形式に整形されたtsvファイルを想定しています。
テキスト部分は一行にまとめて、ラベルは元コーパスの分類を1~9の数字に変換します。
それ以外の部分は削除します。headerとindexも不要です。

例：

  ```
オールドナムコゲームの世界をパックマンが飛び回る？！（以下略）\t7     
  ```

同じく`transformers/src/transformers/data/processors/glue.py`にタスクの説明を追加します。

```python
glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "livedoor": 9, #add
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "livedoor": LivedoorProcessor, #add
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "komachi": "classification",
    "livedoor": "classification", #add
}
```

次に、`transformers/examples/run_glue.py`に以下のように修正します。

先ほど追加したprocessorを反映するためtransformersをimportする前に以下のPathを追加。

```python
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
```
`transformers/src/transformers/data/metrics/__init__.py`に評価用のメトリクスを追加します。

```python
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "livedoor":                   #add
        return {"acc": acc_and_f1(preds, labels)}   #add
    else:
        raise KeyError(task_name)
```
```python
    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        # f1 = f1_score(y_true=labels, y_pred=preds)
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro') #add
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }
```

準備が終わりましたら、以下コマンドでFine-Tuningを実施します。

``` bash
export task_name=livedoor

python transformers/examples/run_glue.py \
    --task_name $task_name \
    --do_train \
    --do_eval \
    --data_dir "$DATA_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --model_type distilbert \
    --model_name_or_path "$MODEL_PATH" \
    --max_seq_length 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 100 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
```
ここで：
```
DATA_PATH       Livedoor Newsコーパスの保存パス
OUTPUT_PATH     FineTunedモデルの保存パス
MODEL_PATH      Pretrainedモデルの保存パス
```

Fine-Tuningされたモデルと、評価結果は$OUTPUT_PATHで生成されることを確認できます。

## おわりに
日本語版DistilBERT事前学習モデルの学習と利用方法を紹介しました。弊社プロジェクト（分類タスク）においては、モデルサイズがBERT-baseの800MBから280MB程度に抑えられるとともに、BERT-baseに比べて9割ほどの精度が得られたため、推論の高速化やモバイルへのデプロイにおいてはかなり実用的と思います。