import datetime
import random
# 计算模型运行时间
import time
import sys
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
# from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
sys.path.append('../')
from work.optimization import AdamW, get_linear_schedule_with_warmup
from work.tokenization_bert import BertTokenizer
# sys.path.append('../')
from eval import *
from modeling_bert_mod_q_layer import BertForMaskedLM
print('modeling_bert_mod_q_layer')
# 设定超参数
SEED = 124
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 8
LEARNING_RATE = 2e-5
EPSILON = 1e-8
regular_alpha = 1e-6
epochs = 20 
use_cuda = "cuda:1"
TEST_SIZE = 0.001
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYHTONHASHSEED"] = str(SEED)
torch.cuda.manual_seed(SEED)


correct_path = '../data/train_data/280k_correct_train_true.txt' 
mistake_path='../data/train_data/280k_mistake_train_true.txt'


device = torch.device(use_cuda if torch.cuda.is_available() else "cpu")
print('device:',device)

model_save_path='../model/bibert-q-layer/'
if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
print('model_save_path:'+model_save_path)


# 读取文件
def readfile(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
        return content


# 将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
def convert_text_to_token(tokenizer, sentence, limit_size=126):
    tokens = tokenizer.encode(sentence[:limit_size])  # 直接截断
    if len(tokens) < limit_size + 2:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


# 建立attention_mask
# attention_masks,在一个文本中，如果是pad符号则是0，否则就是1
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0 ) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks


def get_data_loader(tokenizer, mistake_path, correct_path):
    print('load')
    sentences, targets = readfile(mistake_path)[0:], readfile(correct_path)[0:]  # 读取文本
    input_ids = [convert_text_to_token(tokenizer, '。'+sen) for sen in sentences]  # 转为ids
    targets_ids = [convert_text_to_token(tokenizer,'。'+ sen) for sen in targets]
    input_tokens_tenosr = torch.tensor(input_ids)  # 转为tensor
    targets_tokens_tensor = torch.tensor(targets_ids)
    # 构造attention_mask
    atten_masks = attention_masks(input_ids)
    attention_mask_tensor = torch.tensor(atten_masks)
    print(input_tokens_tenosr.shape)  # torch.Size([10000, 128])
    # 划分训练集和测试集
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens_tenosr, targets_tokens_tensor,
                                                                            #random_state=SEED,
                                                                            test_size=TEST_SIZE)
    train_attention_tokens, test_attention_tokens, _, _ = train_test_split(attention_mask_tensor, attention_mask_tensor,
                                                                           #random_state=SEED, 
                                                                           test_size=TEST_SIZE)
    # 创建DataLoader,用来取出一个batch的数据
    train_data = TensorDataset(train_inputs, train_attention_tokens, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE_TRAIN)

    # 创建测试集的DataLoader
    test_data = TensorDataset(test_inputs, test_attention_tokens, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE_TEST)

    return train_dataloader, test_dataloader


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间


def iterator(model, train_dataloader, test_dataloader):
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON) #, weight_decay=WEIGHT_DECAY
    # 学习率预热，训练时先从小的学习率开始训练
    # training steps 的数量: [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * epochs
    # 设计 learning rate scheduler.
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)
    for epoch in range(epochs):
        # 训练模型
        train_loss = train(model, train_dataloader, optimizer) #,scheduler
        print(f'epoch-{epoch}-train_loss:', train_loss)
        # 评估模型
        test_loss = evaluate(model, test_dataloader)
        print(f'epoch-{epoch}-test_loss:', test_loss)
        torch.save(model, model_save_path+f'best_pytorch_model_{epoch}.bin')




# 训练模型
def train(model, train_dataloader, optimizer,scheduler = None):
    print('train')
    model.train()
    t0 = time.time()
    r_loss = []
    for step, batch in enumerate(train_dataloader):
        # 每隔40个step 输出一下所用时间.
        if step % 100 == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
            2].long().to(device)


        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = output[0], output[1]

        l1_loss = torch.tensor(0.0,requires_grad = True).to(device)
        # l2_loss = torch.tensor(0.0,requires_grad = True).to(device)
        
        for name,param in model.named_parameters():
            if 'linear' in name and '11' in name:
                l1_loss += torch.norm(param, 1)
                # l2_loss += torch.norm(param, 2)
                
        loss = loss + regular_alpha * l1_loss

        optimizer.zero_grad()
        loss.backward()
        r_loss.append(loss.item())
        clip_grad_norm_(model.parameters(), 1.0)  
        optimizer.step()  # 更新模型参数
        # scheduler.step()  # 更新learning rate

    return np.array(r_loss).mean()


# 评估模型
def evaluate(model, test_dataloader):
    avg_loss = []
    model.eval()  # 表示进入测试模式

    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
            2].long().to(device)
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss, logits = output[0], output[1]
        avg_loss.append(loss.item())
    return np.array(avg_loss).mean()




if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('../model/bert-base-chinese')
    model = BertForMaskedLM.from_pretrained("../model/bert-base-chinese")
    model.to(device)

    train_data_loader, test_data_loader = get_data_loader(tokenizer, mistake_path, correct_path)

    iterator(model, train_data_loader, test_data_loader)

    print('finish the train!')

