# BiBSpell


### Prepare the  Base-BERT Model

1.Click https://huggingface.co/bert-base-chinese

2.Click the download icon in "Files and Versions" to download "pytorch_model.bin"

3.move "pytorch_model.bin" to the path "model/bert-base-chinese"

### How to run?

Requirements:
1.Install Anaconda
2.RUN:

```
conda create -n spelLM python=3.6
conda activate spelLM
conda install pytorch-gpu==1.7.0
conda install scikit-learn==0.23.1
conda install tqdm
pip install transformers==3.0.0
unzip -d data/train_data/ data/train_data/280k_mistake_train_true.zip
unzip -d data/train_data/ data/train_data/280k_correct_train_true.zip
```


-----------------------------------------------------------------------
The training are divided into three steps：

**STEP1: Fine-tune the classifier model**
RUN:

```bash
cd train
nohup python -u fine_tune_classifier.py>log_tune_classifier.txt 2>&1 &
```

**STEP2: Train Our Model**
RUN:

```bash
nohup python -u train.py>log/log_train.txt 2>&1 &  
```

(Model will be saved in "model/bibert-q-layer")

**STEP3: Evaluate Our Model**

The official tool has been saved at "test_out/sighan15csc.jar"

RUN:

```bash
cd test_out
java -jar sighan15csc.jar -i sighan15_pre_17.txt -t sighan15_cor_17.txt
```

