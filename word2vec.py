# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
import math
import random
import time
import threading
import sys

MAX_STRING = 100
EXP_TABLE_SIZE = 1000
MAX_EXP = 6
MAX_SENTENCE_LENGTH = 1000
MAX_CODE_LENGTH = 40


class vocab_word(object):
    def __init__(self, word, cn):
        self.word = word
        self.cn = cn
        self.point = [None] * MAX_CODE_LENGTH
        self.code = [None] * MAX_CODE_LENGTH
        self.codelen = 0

    def __lt__(self, other): # override <操作符
        if self.cn > other.cn:
            return True
        return False


train_file = 'text8'
output_file = None
save_vocab_file = ''
read_vocab_file = ''
vocab = None
text = []
text_len = 0
binary = 0
cbow = 1
debug_mode = 2
window = 5
min_count = 5
num_threads = 12
#min_reduce = 1
word2id = {}

vocab_max_size = 1000
vocab_size = 0
layer1_size = 100
train_words = 0
word_count_actual = 0
iter = 5
file_size = 0
classes = 0
alpha = 0.025
starting_alpha = 0.
sample = 1e-3
syn0, syn1, syn1neg = [], [], []
expTable = None

start = None

hs = 0
negative = 5
table_size = 100000000
table = []


def InitUnigramTable():
    global table
    table = [0] * table_size

    train_words_pow = 0.
    power = 0.75
    for a in range(vocab_size):
        train_words_pow += vocab[a].cn ** power

    i = 0
    d1 = vocab[i].cn ** power / train_words_pow
    for a in range(table_size):
        table[a] = i
        if a / table_size > d1:
            i += 1
            d1 += vocab[i].cn ** power / train_words_pow
        if i >= vocab_size:
            i = vocab_size - 1


def SearchVocab(word):
    if word in word2id:
        return word2id[word]
    else:
        return -1


def AddWordToVocab(word, cn):
    tmp = vocab_word(word, cn)
    vocab.append(tmp)


def LearnVocabFromTrainFile():
    global vocab_size
    global text
    global text_len
    global word2id
    global train_words
    t0 = time.time()
    with open(train_file) as f:
        text = f.readline().split()
        text_len = len(text)
    # 排序，过滤，加入词表
    AddWordToVocab('</s>', 0)
    for word, cn in Counter(text).most_common():
        if cn >= min_count:
            AddWordToVocab(word, cn)
    vocab_size = len(vocab)

    word2id = {}
    train_words = 0
    for a in range(len(vocab)):
        word2id[vocab[a].word] = a
        train_words += vocab[a].cn
    print("LearnVocabFromTrainFile cost time:", time.time() - t0)
    print('vocab_size', vocab_size)
    print('words in train file', train_words)


def CreateBinaryTree():
    global vocab
    count, binary, parent_node = [0] * (vocab_size*2+1), [0] * (vocab_size*2+1), [0] * (vocab_size*2+1)
    for a in range(vocab_size):
        count[a] = vocab[a].cn
    for a in range(vocab_size, vocab_size*2):
        count[a] = int(1e15)
    pos1, pos2 = vocab_size - 1, vocab_size
    min1i, min2i = None, None
    for a in range(vocab_size-1):
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min1i = pos1
                pos1 -= 1
            else:
                min1i = pos2
                pos2 += 1
        else:
            min1i = pos2
            pos2 += 1
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min2i = pos1
                pos1 -= 1
            else:
                min2i = pos2
                pos2 += 1
        else:
            min2i = pos2
            pos2 += 1
        count[vocab_size + a] = count[min1i] + count[min2i]
        parent_node[min1i] = vocab_size + a
        parent_node[min2i] = vocab_size + a
        binary[min2i] = 1
    code = [0] * MAX_CODE_LENGTH
    point = [0] * MAX_CODE_LENGTH
    for a in range(vocab_size):
        b = a
        i = 0
        while True:
            code[i] = binary[b]
            point[i] = b
            i += 1
            b = parent_node[b]
            if b == vocab_size * 2 - 2:
                break
        vocab[a].codelen = i
        vocab[a].point[0] = vocab_size - 2
        for b in range(i):
            vocab[a].code[i-b-1] = code[b]
            vocab[a].point[i-b] = point[b] - vocab_size


def InitNet():
    global syn0, syn1, syn1neg
    syn0 = ((np.random.rand(vocab_size * layer1_size) - 0.5) / layer1_size).tolist() # TODO: [0,1) to [0,1] 是否有区别
    if hs == 1:
        syn1 = [0] * (vocab_size * layer1_size)
    if negative > 0:
        syn1neg = [0] * (vocab_size * layer1_size)
    CreateBinaryTree()


def TrainModelThread(id):
    global word_count_actual
    global alpha
    sentence_length = 0
    sentence_position = 0
    word_count = 0
    last_word_count = 0
    sen = [None] * (MAX_SENTENCE_LENGTH+1)
    local_iter = iter
    eof = 0
    neu1, neu1e = [0] * layer1_size, [0] * layer1_size
    pointer = text_len // num_threads * id
    while True:
        if word_count - last_word_count > 10000:
            word_count_actual += word_count - last_word_count
            last_word_count = word_count
            if debug_mode > 1:
                now = time.time()
                print('\nAlpha:{0}  Progress: {1}%  Words/thread: {2}k  Cost time: {3}'.format(alpha, word_count_actual / (iter * train_words + 1) * 100, word_count_actual / ((now - start + 1) * 1000), now-start))
                #sys.stdout.flush()
            alpha = starting_alpha * (1 - word_count_actual / (iter * train_words + 1))
            if alpha < starting_alpha * 0.0001:
                alpha = starting_alpha * 0.0001
        if sentence_length == 0:
            while True:
                if pointer >= text_len:
                    break
                word = SearchVocab(text[pointer])
                pointer += 1
                if word == -1:
                    continue
                word_count += 1
                if word == 0:
                    break
                if sample > 0:
                    ran = (math.sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn
                    next_random = random.uniform(0, 1)
                    if ran < next_random:
                        continue
                sen[sentence_length] = word
                sentence_length += 1
                if sentence_length > MAX_SENTENCE_LENGTH:
                    break
            sentence_position = 0
        if pointer >= text_len or word_count > train_words // num_threads:
            print('thread ', id, 'finish iter ', iter-local_iter)
            if id == 0:
                with open(output_file+'iter'+str(iter-local_iter), 'w') as fw:
                    fw.write("{0} {1}\n".format(vocab_size, layer1_size))
                    for a in range(vocab_size):
                        fw.write("{0} ".format(vocab[a].word))
                        for b in range(layer1_size):
                            fw.write("{0} ".format(syn0[a * layer1_size + b]))
                        fw.write('\n')
            word_count_actual += word_count - last_word_count
            local_iter -= 1
            if local_iter == 0:
                break
            word_count = 0
            last_word_count = 0
            sentence_length = 0
            pointer = text_len // num_threads * id
            continue
        word = sen[sentence_position]
        if word == -1:
            continue
        for c in range(layer1_size):
            neu1[c] = 0
        for c in range(layer1_size):
            neu1e[c] = 0
        b = random.randint(0, window-1)
        if cbow:
            cw = 0
            for a in range(b, window * 2 + 1 - b):
                if a != window:
                    c = sentence_position - window + a
                    if c < 0:
                        continue
                    if c >= sentence_length:
                        continue
                    last_word = sen[c]
                    if last_word == -1:
                        continue
                    for c in range(layer1_size):
                        neu1[c] += syn0[c + last_word * layer1_size]
                    cw += 1
            if cw:
                for c in range(layer1_size):
                    neu1[c] /= cw
                if hs:
                    for d in range(vocab[word].codelen):
                        f = 0
                        l2 = vocab[word].point[d] * layer1_size
                        f += np.sum(np.multiply(np.array(neu1), np.array(syn1[l2:l2+layer1_size])))
                        #for c in range(layer1_size):
                        #    f += neu1[c] * syn1[c + l2]
                        if f <= -MAX_EXP:
                            continue
                        elif f >= MAX_EXP:
                            continue
                        else:
                            f = expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE // MAX_EXP // 2))]
                        g = (1 - vocab[word].code[d] - f) * alpha
                        for c in range(layer1_size):
                            neu1e[c] += g * syn1[c + l2]
                        for c in range(layer1_size):
                            syn1[c + l2] += g * neu1[c]
                if negative > 0:
                    for d in range(negative + 1):
                        if d == 0:
                            target = word
                            label = 1
                        else:
                            target = table[random.randint(0, table_size-1)]
                            if target == 0:
                                target = random.randint(0, vocab_size-2) + 1
                            if target == word:
                                continue
                            label = 0
                        l2 = target * layer1_size
                        f = 0
                        f += np.sum(np.multiply(np.array(neu1), np.array(syn1neg[l2:l2+layer1_size])))
                        #for c in range(layer1_size):
                        #    f += neu1[c] * syn1neg[c + l2]
                        if f > MAX_EXP:
                            g = (label - 1) * alpha
                        elif f < -MAX_EXP:
                            g = (label - 0) * alpha
                        else:
                            g = (label - expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE // MAX_EXP // 2))]) * alpha
                        for c in range(layer1_size):
                            neu1e[c] += g * syn1neg[c + l2]
                        for c in range(layer1_size):
                            syn1neg[c + l2] += g * neu1[c]
                for a in range(b, window * 2 + 1 - b):
                    if a != window:
                        c = sentence_position - window + a
                        if c < 0:
                            continue
                        if c >= sentence_length:
                            continue
                        last_word = sen[c]
                        if last_word == -1:
                            continue
                        for c in range(layer1_size):
                            syn0[c + last_word * layer1_size] += neu1e[c]
        else:
            for a in range(b, window * 2 + 1 - b):
                if a != window:
                    c = sentence_position - window + a
                    if c < 0:
                        continue
                    if c >= sentence_length:
                        continue
                    last_word = sen[c]
                    if last_word == -1:
                        continue
                    l1 = last_word * layer1_size
                    for c in range(layer1_size):
                        neu1e[c] = 0
                    if hs:
                        for d in range(vocab[word].codelen):
                            f = 0
                            l2 = vocab[word].point[d] * layer1_size
                            f += np.sum(np.multiply(np.array(syn0[l1:l1+layer1_size]), np.array(syn1[l2:l2+layer1_size])))
                            #for c in range(layer1_size):
                            #    f += syn0[c + l1] * syn1[c + l2]
                            if f <= -MAX_EXP:
                                continue
                            elif f >= MAX_EXP:
                                continue
                            else:
                                f = expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE // MAX_EXP // 2))]
                            g = (1 - vocab[word].code[d] - f) * alpha
                            for c in range(layer1_size):
                                neu1e[c] += g * syn1[c + l2]
                            for c in range(layer1_size):
                                syn1[c + l2] += g * syn0[c + l1]
                    if negative > 0:
                        for d in range(negative + 1):
                            if d == 0:
                                target = word
                                label = 1
                            else:
                                target = table[random.randint(0, table_size - 1)]
                                if target == 0:
                                    target = random.randint(0, vocab_size - 2) + 1
                                if target == word:
                                    continue
                                label = 0
                            l2 = target * layer1_size
                            f = 0
                            f += np.sum(np.multiply(np.array(syn0[l1:l1+layer1_size]), np.array(syn1neg[l2:l2+layer1_size])))
                            #for c in range(layer1_size):
                            #    f += syn0[c + l1] * syn1neg[c + l2]
                            if f > MAX_EXP:
                                g = (label - 1) * alpha
                            elif f < -MAX_EXP:
                                g = (label - 0) * alpha
                            else:
                                g = (label - expTable[int((f + MAX_EXP) * (EXP_TABLE_SIZE // MAX_EXP // 2))]) * alpha
                            for c in range(layer1_size):
                                neu1e[c] += g * syn1neg[c + l2]
                            for c in range(layer1_size):
                                syn1neg[c + l2] += g * syn0[c + l1]
                    for c in range(layer1_size):
                        syn0[c + l1] += neu1e[c]
        sentence_position += 1
        if sentence_position >= sentence_length:
            sentence_length = 0
            continue


def TrainModel():
    global starting_alpha
    global start
    print("Starting training using file %s" % train_file)
    starting_alpha = alpha
    LearnVocabFromTrainFile()
    if output_file is None:
        return
    InitNet()
    if negative > 0:
        InitUnigramTable()
    start = time.time()
    thread_list = []
    for id in range(num_threads):
        thread = threading.Thread(target=TrainModelThread, args=(id,))
        thread_list.append(thread)
    thread = None
    for thread in thread_list:
        thread.setDaemon(True)
        thread.start()
    thread.join()
    with open(output_file, 'w') as fw:
        if classes == 0:
            fw.write("{0} {1}\n".format(vocab_size, layer1_size))
            for a in range(vocab_size):
                fw.write("{0} ".format(vocab[a].word))
                for b in range(layer1_size):
                    fw.write("{0} ".format(syn0[a * layer1_size + b]))
                fw.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--size', type=int, default=200)
    parser.add_argument('--train', type=str, default='text8')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--read', type=str, default=None)
    parser.add_argument('--debug', type=int, default=None)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--cbow', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--output', type=str, default='vectors_output_window4')
    parser.add_argument('--window', type=int, default=4)
    parser.add_argument('--sample', type=float, default=1e-4)
    parser.add_argument('--hs', type=int, default=0)
    parser.add_argument('--negative', type=int, default=25)
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--iter', type=int, default=5)
    parser.add_argument('--mincount', type=int, default=5)
    parser.add_argument('--classes', type=int, default=0)
    args = parser.parse_args()
    if args.size is not None:
        layer1_size = args.size
    if args.train is not None:
        train_file = args.train
    if args.save is not None:
        save_vocab_file = args.save
    if args.read is not None:
        read_vocab_file = args.read
    if args.debug is not None:
        debug_mode = args.debug
    if args.binary is not None:
        binary = args.binary
    if args.cbow is not None:
        cbow = args.cbow
    if cbow:
        alpha = 0.05
    if args.alpha is not None:
        alpha = args.alpha
    if args.output is not None:
        output_file = args.output
    if args.window is not None:
        window = args.window
    if args.sample is not None:
        sample = args.sample
    if args.hs is not None:
        hs = args.hs
    if args.negative is not None:
        negative = args.negative
    if args.threads is not None:
        num_threads = args.threads
    if args.iter is not None:
        iter = args.iter
    if args.mincount is not None:
        min_count = args.mincount
    if args.classes is not None:
        classes = args.classes
    vocab = []
    expTable = [None] * (EXP_TABLE_SIZE + 1)
    for i in range(EXP_TABLE_SIZE):
        expTable[i] = math.exp((i / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        expTable[i] = expTable[i] / (expTable[i] + 1)
    TrainModel()







'''

LearnVocabFromTrainFile()
InitNet()
#InitUnigramTable()

print(vocab[-1].word, vocab[-1].cn, vocab[-1].point, vocab[-1].code, vocab[-1].codelen)
for i in range(4):
    print(vocab[i].word, vocab[i].cn, vocab[i].point, vocab[i].code, vocab[i].codelen)

'''

