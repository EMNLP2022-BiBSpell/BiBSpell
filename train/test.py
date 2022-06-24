import random
import sys
import datetime
import re
import time
import os

import numpy as np
import torch
sys.path.append('../')
from work.tokenization_bert import BertTokenizer
from modeling_bert_mod_q_layer import BertForMaskedLM
print('modeling_bert_mod_q_layer')


SEED = 123
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
EPSILON = 1e-8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYHTONHASHSEED"] = str(SEED)
torch.cuda.manual_seed(SEED)


# 读取文件
def readfile(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
        return content


# attention_masks,在一个文本中，如果是pad符号则是0，否则就是1
# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks


class_model_path = '../model/bert-base-chinese'

tokenizer = BertTokenizer.from_pretrained(class_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 预测
def predict_file_one(file_path, out_path, correct_path, model):
    with open(file_path, 'r') as f, open(out_path, 'w') as p, open(correct_path, 'r') as c:
        count = 0
        lines = f.readlines()
        c_lines = c.readlines()
        print("句子数量：",len(lines))
        for i in range(len(lines)):
            # print('sen:', sen)
            sen = lines[i]
            # c_line = c_lines[i]

            # print('原始：', sen)
            # print('正确：', c_line)
            ids = tokenizer.encode('。' + sen)
            b_input_ids = torch.tensor([ids]).to(device)
            logits = model(b_input_ids,
                           attention_mask=None)
            prediction_scores = logits[0]
            pre = torch.softmax(prediction_scores[0], -1)
            top_info = torch.topk(pre, k=5)
            scores = top_info[0]
            ids = top_info[1].squeeze()
            sen_new = []
            for i in range(2, ids.size()[0] - 1):
                predicted_tokens = tokenizer.convert_ids_to_tokens(ids[i])
                # print(predicted_tokens)
                # 取top1的结果，并写到文件中
                # 取top1的结果，并写到文件中
                sen_new.append(predicted_tokens[0])

            p.write(''.join(sen_new) + '\n')
            count += 1
            print(f'{count}:' + ''.join(sen_new) + '\n')
            # print('*' * 50)
        print('a')


def strQ2B(ustring):
    """中文特殊符号转英文特殊符号"""
    #中文特殊符号批量识别

    pattern = re.compile('[，。：“”【】《》？；、（）‘’『』「」﹃﹄〔〕—·]')
    
    #re.compile: 编译一个正则表达式模式，返回一个模式（匹配模式）对象。
    #[...]用于定义待转换的中文特殊符号字符集

    fps = re.findall(pattern, ustring)
    
    #re.findall: 搜索string，以列表形式返回全部能匹配的子串。

    #对有中文特殊符号的文本进行符号替换
    
    if len(fps) > 0:
        ustring = ustring.replace('，', ',')
        ustring = ustring.replace('。', '.')
        ustring = ustring.replace('：', ':')
        ustring = ustring.replace('“', '"')
        ustring = ustring.replace('”', '"')
        ustring = ustring.replace('【', '[')
        ustring = ustring.replace('】', ']')
        ustring = ustring.replace('《', '<')
        ustring = ustring.replace('》', '>')
        ustring = ustring.replace('？', '?')
        ustring = ustring.replace('；', ':')
        ustring = ustring.replace('、', ',')
        ustring = ustring.replace('（', '(')
        ustring = ustring.replace('）', ')')
        ustring = ustring.replace('‘', "'")
        ustring = ustring.replace('’', "'")
        ustring = ustring.replace('’', "'")
        ustring = ustring.replace('『', "[")
        ustring = ustring.replace('』', "]")
        ustring = ustring.replace('「', "[")
        ustring = ustring.replace('」', "]")
        ustring = ustring.replace('﹃', "[")
        ustring = ustring.replace('﹄', "]")
        ustring = ustring.replace('〔', "{")
        ustring = ustring.replace('〕', "}")
        ustring = ustring.replace('—', "-")
        ustring = ustring.replace('·', ".")
    
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:                                #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):   #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring
    

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间

if __name__ == '__main__':
    print('进入预测阶段：')
    f_id = 5
    EPOCH = 20

    for epoch_id in range(0, EPOCH):
        model_path=f'../model/bibert-q-layer/best_pytorch_model_{epoch_id}.bin'
        test_out_path=f'../test_out/sighan1{f_id}_output{epoch_id}.txt'
        model = torch.load(model_path)  

        print('epoch:::::::::', epoch_id)
        a = time.time()

        predict_file_one(file_path, test_out_path, correct_path, model)
        print(time.time() - a)

        pre_file=test_out_path
        truth_file=f'../data/test_data/sighan1{f_id}_test/sig_1{f_id}_correct_test.txt'#正确的标签句
        mistake_file=f'../data/test_data/sighan1{f_id}_test/sig_1{f_id}_mistake_test.txt'#原始错误的句子

        #输出
        out_pre=f'../test_out/sighan1{f_id}_pre_{epoch_id}.txt'
        out_cor=f'../test_out/sighan1{f_id}_cor_{epoch_id}.txt'

        with open(pre_file) as p_f,open(truth_file) as t_f,open(mistake_file) as m_f, open(out_pre,'w') as out_pre_f,open(out_cor,'w') as out_cor_f:
            m_lines,p_lines,c_lines=m_f.readlines()[0:],p_f.readlines()[0:],t_f.readlines()[0:]
            for i in range(len(p_lines)):
                # print(p_lines[i])
                m_line=strQ2B(m_lines[i].strip())
                p_line=strQ2B(p_lines[i].strip())
                c_line=strQ2B(c_lines[i].strip())
                m_line = re.sub(' ', '', m_line)
                p_line = re.sub('#', '', p_line)
                p_line = re.sub('\[UNK\]', '.', p_line)
                p_line = re.sub('\[SEP\]', '.', p_line)
                p_line = re.sub('\[PAD\]', '', p_line)
                # p_line = re.sub('..', '.', p_line)
                p_line = re.sub('~~~', '!', p_line)
                p_line = re.sub('~~', '!', p_line)
                p_line = re.sub('~', '!', p_line)
                p_line = re.sub(' ', '', p_line)
                if len(m_line) !=len(p_line):
                    print('I:',i)
                    print(p_lines[i])
                    print(m_line)
                    print(p_line)
                    print(c_line)
                    print('*'*50)

                    continue

                sen_id=str(10+i)+'A'
                pre_list=[]
                cor_list=[]
                pre_list.append(sen_id)
                cor_list.append(sen_id)
                add_pre_0_flag=True
                add_cor_0_flag=True
                for j in range(len(p_line)):
                    if p_line[j].islower() and p_line[j].upper() == c_line[j]:
                        continue
                    if p_line[j] !=m_line[j]:#说明检测到错误
                        add_pre_0_flag=False
                        pre_list.append(str(j+1))
                        pre_list.append(p_line[j])
                    if c_line[j] !=m_line[j]:#说明本来就有错误
                        add_cor_0_flag=False
                        cor_list.append(str(j+1))
                        cor_list.append(c_line[j])
                if add_pre_0_flag:
                    pre_list.append(str(0))
                if add_cor_0_flag:
                    cor_list.append(str(0))
                

                out_pre_f.write(', '.join(pre_list)+'\n')   
                out_cor_f.write(', '.join(cor_list)+'\n')   