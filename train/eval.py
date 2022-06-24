# -*- coding: GBK -*-
import re


def strQ2B(ustring):
    """�����������תӢ���������"""
    # ���������������ʶ��

    pattern = re.compile('[������������������������������������������롲������]')

    # re.compile: ����һ��������ʽģʽ������һ��ģʽ��ƥ��ģʽ������
    # [...]���ڶ����ת����������������ַ���

    fps = re.findall(pattern, ustring)

    # re.findall: ����string�����б���ʽ����ȫ����ƥ����Ӵ���

    # ��������������ŵ��ı����з����滻

    if len(fps) > 0:
        ustring = ustring.replace('��', ',')
        ustring = ustring.replace('��', '.')
        ustring = ustring.replace('��', ':')
        ustring = ustring.replace('��', '"')
        ustring = ustring.replace('��', '"')
        ustring = ustring.replace('��', '[')
        ustring = ustring.replace('��', ']')
        ustring = ustring.replace('��', '<')
        ustring = ustring.replace('��', '>')
        ustring = ustring.replace('��', '?')
        ustring = ustring.replace('��', ':')
        ustring = ustring.replace('��', ',')
        ustring = ustring.replace('��', '(')
        ustring = ustring.replace('��', ')')
        ustring = ustring.replace('��', "'")
        ustring = ustring.replace('��', "'")
        ustring = ustring.replace('��', "'")
        ustring = ustring.replace('��', "[")
        ustring = ustring.replace('��', "]")
        ustring = ustring.replace('��', "[")
        ustring = ustring.replace('��', "]")
        ustring = ustring.replace('��', "[")
        ustring = ustring.replace('��', "]")
        ustring = ustring.replace('��', "{")
        ustring = ustring.replace('��', "}")
        ustring = ustring.replace('��', "-")
        ustring = ustring.replace('��', ".")

    """ȫ��ת���"""
    # ת��˵����
    # ȫ���ַ�unicode�����65281~65374 ��ʮ������ 0xFF01 ~ 0xFF5E��
    # ����ַ�unicode�����33~126 ��ʮ������ 0x21~ 0x7E��
    # �ո�Ƚ����⣬ȫ��Ϊ 12288��0x3000�������Ϊ 32��0x20��
    # ���ո��⣬ȫ��/��ǰ�unicode����������˳�����Ƕ�Ӧ�ģ���� + 0x7e= ȫ�ǣ�,���Կ���ֱ��ͨ����+-��������ǿո����ݣ��Կո񵥶�����

    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # ȫ�ǿո�ֱ��ת��
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # ȫ���ַ������ո񣩸��ݹ�ϵת��
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def cal_pre(n):
    detection_chars = []
    # correctedly_chars=[]
    # detectedly_chars=[]

    all_errors = 0  # ���д����ַ�
    all_correction = 0  # ���о������ַ�
    all_detection = 0  # ���м����ַ�
    all_correctedly = 0  # ������ȷ�������ַ�
    all_detectedly = 0  # ������ȷ�����ַ�

    # mask_new_{file}.txt
    with open('../data/sig_15_mistake_test_gcn.txt', 'r') as m, open('trian_ig_test_pre.txt', 'r') as p, open(
            '../data/sig_15_correct_test_gcn.txt', 'r') as c:
        m_lines, p_lines, c_lines = m.readlines()[0:n], p.readlines(), c.readlines()[0:n]
        count = 0
        for i in range(len(m_lines)):

            #         m_lines[i] = re.sub('||', '', m_lines[i])
            m_line = strQ2B(m_lines[i].strip())
            c_line = strQ2B(c_lines[i].strip())
            p_line = strQ2B(p_lines[i].strip())
            m_line = re.sub(' ', '', m_line)
            p_line = re.sub('#', '', p_line)
            p_line = re.sub('\[UNK\]', '.', p_line)

            p_line = re.sub('\[SEP\]', '.', p_line)
            #         p_line = re.sub('||', '', p_line)
            p_line = re.sub(' ', '', p_line)
            if len(m_line) != len(p_line):
                print('I:', i)
                print(m_line)
                print(p_line)
                print(c_line)
                print('*' * 50)
                continue
            #         print(m_line)
            #         print(c_line)
            #         print(p_line)
            count += 1
            for j in range(len(m_line)):
                if len(re.findall(r'[a-zA-Z��-�ڣ�-��]+', p_line[j])) != 0 or len(
                        re.findall(r'[a-zA-Z��-�ڣ�-��]+', m_line[j])) != 0:
                    continue
                if len(re.findall(r'[?.,!]+', p_line[j])) != 0:
                    continue
                if j == 0:
                    if p_line[j] == "." or p_line[j] == ",":
                        continue
                if m_line[j] != c_line[j]:  # ԭʼ�ĺ���ȷ�Ĳ�һ������Ϊ�Ǵ����
                    all_errors += 1
                if m_line[j] != p_line[j]:  # ԭʼ�ַ���Ԥ��Ĳ�ͬ����˵���Ǽ����ַ�
                    all_detection += 1
                    if m_line[j] != c_line[j]:  # �����д��Ҽ�⵽�д�
                        all_detectedly += 1
                if m_line[j] != c_line[j]:
                    if m_line[j] != p_line[j]:  # �����д�ĵط�,������
                        all_correction += 1
                    if m_line[j] != p_line[j] and p_line[j] == c_line[j]:  # �����д�ĵط��������Ҿ�������
                        all_correctedly += 1

    # �õ����еĴ����ַ�������
    # all_errors=491
    # print('����:',''.join(detection_chars))
    # print('��ȷ����:',''.join(detectedly_chars))
    # print('��ȷ������:',''.join(correctedly_chars))
    print('count', count)
    print('all_errors', all_errors)
    print('all_detection', all_detection)
    print('all_detectedly', all_detectedly)
    print('all_correction', all_correction)
    print('all_correctedly', all_correctedly)

    check_p = round(all_detectedly / all_detection, 5)
    check_r = round(all_detectedly / all_errors, 5)
    check_F1 = round(2 * (check_p * check_r) / (check_p + check_r), 5)
    print(f'��⾫ȷ��: ��ȷ����/���м������ַ� char_p={all_detectedly}/{all_detection}\t', check_p)
    print(f'����ٻ���: ��ȷ����/���д�����ַ�   char_p={all_detectedly}/{int(all_errors)}\t', check_r)
    print(f'����F1ֵ:', check_F1)

    correct_p = round(all_correctedly / all_correction, 5)
    correct_r = round(all_correctedly / all_errors, 5)
    correct_F1 = round(2 * (correct_p * correct_r) / (correct_p + correct_r), 5)
    print(f'������ȷ��: ��ȷ������/���о������ַ�  char_p={all_correctedly}/{all_correction}\t', correct_p)
    print(f'�����ٻ���: ��ȷ������/���д�����ַ�  char_p={all_correctedly}/{int(all_errors)}\t', correct_r)
    print(f'������F1ֵ:', correct_F1)
    print('*' * 50)
