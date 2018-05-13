import argparse
import cPickle
import h5py
import os
import numpy as np

import math
from tqdm import tqdm
import multiprocessing
from threading import Thread
from itertools import groupby
from collections import Counter
from nltk.corpus import stopwords  # remove stopwords and too frequent words (in, a, the ..)
from collections import defaultdict, Counter

from util import log


cpu_count = multiprocessing.cpu_count()
num_thread = max(cpu_count - 2, 1)

def str_list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(',')]

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--enwiki_dirs', type=str_list,
                    default='data/preprocessed/enwiki/enwiki_processed_1_4,' \
                            'data/preprocessed/enwiki/enwiki_processed_2_4,' \
                            'data/preprocessed/enwiki/enwiki_processed_3_4,' \
                            'data/preprocessed/enwiki/enwiki_processed_4_4', help=' ')
parser.add_argument('--genome_annotation_dir', type=str,
                    default='data/VisualGenome/annotations', help=' ')
parser.add_argument('--dir_name', type=str,
                    default='data/preprocessed/visualgenome'
                    '/memft_all_new_vocab50_obj3000_attr1000_maxlen10', help=' ')
parser.add_argument('--context_window_size', type=int, default=3,
                    help='window size for extracting context')
parser.add_argument('--preprocessing', type=int, default=0,
                    help='whether to do preprocessing (1) or not (0)')
parser.add_argument('--min_num_word', type=int, default=5, help='min num word in set')
config = parser.parse_args()

config.answer_dict_path = os.path.join(config.dir_name, 'answer_dict.pkl')
answer_dict = cPickle.load(open(config.answer_dict_path, 'rb'))

word2contexts = None

dummy = {
    ') , her <word> was amputated .': 1,
    ') in his <word> , requiring surgery': 1,
    ', and his <word> was mangled so': 1,
    ', and the <word> is extended and': 1,
    ', and their <word> fully extended ,': 1,
    ', arms and <word> were integrated in': 1,
    ', but her <word> remained shorter ,': 1,
    ', chest and <word> of the king': 1,
    ", diana 's <word> bent and exposed": 1,
    ", donnell 's <word> appeared to strike": 1,
    ", henderson 's <word> was broken in": 1,
    ', his entire <word> was amputated .': 1,
    ', losing his <word> and much of': 1,
    ', lost her <word> above the knee': 1,
    ", montalvan 's <word> was amputated above": 1,
    ', neck , <word> , and left': 1,
    ', neck , <word> and left hand': 1,
    ", palmatier 's <word> was crushed in": 1,
    ', pelvis , <word> , and abdomen': 1,
    ', piercing his <word> , and the': 1,
    ', resting his <word> above the knee': 1,
    ', severing his <word> and also severely': 1,
    ", shiva 's <word> is on the": 1,
    ', then her <word> . <unk> <unk>': 1,
    ', to his <word> , abdomen ,': 1,
    ', where his <word> was amputated .': 1,
    ', where his <word> was amputated above': 1,
    ', where the <word> is shown as': 1,
    ', while her <word> is bent high': 1,
    ', while her <word> is outstretched in': 1,
    ', while her <word> sweeps behind it': 1,
    ', while the <word> seems to have': 1,
    ', whose lower <word> had been amputated': 1,
    ', with her <word> folded and the': 1,
    ', with her <word> turned slightly to': 1,
    ', with her <word> wrapped around its': 1,
    '- technical ; <word> and left leg': 1,
    '00 with a <word> injury . <unk>': 1,
    '0000 , his <word> was badly wounded': 1,
    '0000 without a <word> and started to': 1,
    ': baskets with <word> , his back': 1,
    '; thumb of <word> remains unmoved during': 1,
    "; tityus ' <word> lies flat while": 1,
    '<unk> <unk> aungles <word> was malformed at': 1,
    '<unk> <unk> her <word> fractured from three': 1,
    '<unk> <unk> her <word> is in mooladhara': 1,
    '<unk> <unk> her <word> seems to be': 1,
    '<unk> <unk> her <word> was amputated when': 1,
    '<unk> <unk> her <word> was the most': 1,
    '<unk> <unk> her <word> would eventually be': 1,
    '<unk> <unk> his <word> folded over the': 1,
    '<unk> <unk> his <word> is folded over': 1,
    '<unk> <unk> his <word> is grey matter': 1,
    '<unk> <unk> his <word> required amputation .': 1,
    '<unk> <unk> his <word> was amputated above': 1,
    '<unk> <unk> his <word> was amputated after': 1,
    '<unk> <unk> his <word> was amputated below': 1,
    '<unk> <unk> his <word> was amputated following': 1,
    '<unk> <unk> its <word> has a small': 1,
    '<unk> <unk> the <word> was bent under': 1,
    "<unk> adonis ' <word> looks to be": 1,
    "<unk> dodi 's <word> is completely off": 1,
    '<unk> dragging his <word> , benito chose': 1,
    "<unk> elsey 's <word> was amputated in": 1,
    "<unk> fang 's <word> was amputated at": 1,
    '<unk> his transplanted <word> had to be': 1,
    '<unk> his twisted <word> injury from black': 1,
    '<unk> its front <word> is elevated off': 1,
    '<unk> losing his <word> as result of': 1,
    '<unk> losing his <word> below the knee': 1,
    '<unk> on his <word> , a benign': 1,
    "<unk> rehm 's <word> is amputated and": 1,
    "<unk> sampson 's <word> was broken ,": 1,
    "a fully formed <word> '' , and": 1,
    'a injury in <word> which kept him': 1,
    'a partially amputated <word> , is bitter': 1,
    'a survivor whose <word> had been chopped': 1,
    'abdomen and the <word> . <unk> <unk>': 1,
    'abductor in his <word> which he picked': 1,
    'abscess on her <word> following a suspected': 1,
    'achilles in his <word> while competing on': 1,
    'acl in her <word> meaning she spent': 1,
    'additionally , the <word> was broken .': 1,
    'adductor in his <word> during a training': 1,
    'after breaking his <word> while filming ``': 1,
    'after sustaining a <word> injury , ruling': 1,
    'again after his <word> is paralysed by': 1,
    'against his proper <word> . <unk> <unk>': 1,
    'also suffered a <word> injury against the': 1,
    'among them her <word> and a detachable': 1,
    'amputate his gangrenous <word> in september 0000': 1,
    'amputation in his <word> when he was': 1,
    'amputation of her <word> . <unk> <unk>': 1,
    'amputation of her <word> below the knee': 1,
    'amputation of her <word> due to a': 1,
    'amputation of her <word> in 0000 after': 1,
    'amputation of his <word> , the doctor': 1,
    'amputation of his <word> . <unk> <unk>': 5,
    'amputation of his <word> above the knee': 1,
    'amputation of his <word> and the loss': 1,
    'an articulated complete <word> and foot .': 1,
    'an elephant with <word> hanging and left': 1,
    'and a twisted <word> which had to': 1,
    'and bend their <word> sharply backwards .': 1,
    'and breaking his <word> in two places': 1,
    'and broke his <word> , giving the': 1,
    'and broke his <word> at genola (': 1,
    'and broke his <word> in an injury': 1,
    'and broke his <word> in the subsequent': 1,
    'and crippled his <word> . <unk> <unk>': 1,
    "and crushed his <word> '' . <unk>": 1,
    'and folding her <word> . <unk> <unk>': 1,
    'and had his <word> amputated below the': 1,
    'and had his <word> amputated from approximately': 1,
    'and has his <word> crossed over his': 1,
    'and hence his <word> is the lead': 1,
    'and losing his <word> . <unk> <unk>': 1,
    'and lost his <word> . <unk> <unk>': 1,
    'and minus his <word> , awakens in': 1,
    'and severed his <word> through the knee': 1,
    'and the lower <word> below the knee': 1,
    'and the lower <word> has broken off': 1,
    'and with the <word> totally incapacitated ,': 1,
    'arc of the <word> , and the': 1,
    'arm and the <word> . <unk> <unk>': 1,
    'arms or his <word> . <unk> <unk>': 1,
    'arrows above his <word> . <unk> <unk>': 1,
    'atrophy of his <word> . <unk> <unk>': 1,
    'awakens without his <word> and under the': 1,
    'a\xc3\x9fmann broke his <word> . <unk> <unk>': 1,
    'back her upper <word> . <unk> <unk>': 1,
    'back of her <word> . <unk> <unk>': 1,
    'back of his <word> . <unk> <unk>': 1,
    'back while the <word> is stretched on': 1,
    'backpack and his <word> is bent to': 1,
    'backward , the <word> of his pants': 1,
    'bassi in the <word> . <unk> <unk>': 1,
    'bending of the <word> suggested to investigators': 1,
    'birmingham where his <word> was amputated above': 1,
    'birth , his <word> had a deficiency': 1,
    'bomb and his <word> was amputated .': 1,
    'bone in her <word> . <unk> <unk>': 1,
    'bone of his <word> , which required': 1,
    'bones in his <word> in a sprint': 1,
    'bones in his <word> in four places': 1,
    'born with a <word> defect , with': 1,
    'born with her <word> shorter than her': 1,
    'born with her <word> significantly shorter than': 1,
    'born without a <word> and with bones': 1,
    'born without her <word> . <unk> <unk>': 1,
    'both arms , <word> and a graze': 1,
    'bottom of the <word> ; the bullet': 1,
    "boyd 's lower <word> , biting and": 1,
    'brace on his <word> . <unk> <unk>': 1,
    'break in his <word> . <unk> <unk>': 1,
    'breaking both his <word> and left foot': 1,
    'brooker had his <word> amputated when he': 1,
    'busch broke his <word> and fractured his': 1,
    'but lost his <word> after developing compartment': 1,
    'calf of his <word> with a saw': 1,
    'cancer in her <word> , leading to': 1,
    'cancer in her <word> and doctors were': 1,
    'canister in the <word> . <unk> <unk>': 1,
    'captain in his <word> . <unk> <unk>': 1,
    "carl delong 's <word> , and left": 1,
    "carl delong 's <word> . <unk> <unk>": 1,
    'cartilage in his <word> from an injury': 1,
    'cases , the <word> ) and balancing': 2,
    'chileshe broke his <word> in the 00th': 1,
    'chit of the <word> first is taken': 1,
    'contains the severed <word> of a woman': 1,
    'cruise injured his <word> on the london': 1,
    'cut off her <word> . <unk> <unk>': 1,
    'damage in her <word> . <unk> <unk>': 1,
    'damage in his <word> . <unk> <unk>': 1,
    'damage on his <word> . <unk> <unk>': 1,
    'damage to his <word> . <unk> <unk>': 1,
    'damaged ; the <word> supports his weight': 1,
    "deer 's front <word> is raised off": 1,
    'deformity in her <word> . <unk> <unk>': 1,
    'deformity in her <word> which made it': 1,
    'depicted with his <word> in relaxing posture': 2,
    'deputat injured his <word> and decided to': 2,
    'despite this her <word> was significantly shorter': 1,
    'disability in his <word> when he was': 1,
    'disability to his <word> and right foot': 1,
    'discomfort in his <word> . <unk> <unk>': 1,
    'discovered in her <word> . <unk> <unk>': 1,
    'dismembering his lower <word> . <unk> <unk>': 1,
    "diving suit 's <word> by a sharp": 1,
    'domain ball , <word> and left .': 1,
    'down only her <word> which was left': 1,
    'due to a <word> stress fracture he': 1,
    "durst 's lower <word> was amputated in": 1,
    'eight and her <word> at age 00': 1,
    'elbow , her <word> is `` severely': 1,
    'end of his <word> . <unk> <unk>': 1,
    'equestrian rider and <word> amputee spadicia harris': 1,
    'events with a <word> injury . <unk>': 1,
    'eventually had his <word> amputated . <unk>': 1,
    'feeling in his <word> . <unk> <unk>': 1,
    'fell on his <word> . <unk> <unk>': 1,
    'femoris in his <word> . <unk> <unk>': 1,
    'femur in his <word> and three broken': 1,
    'fibula in her <word> and because of': 1,
    'fibula in his <word> , causing him': 1,
    'fibula in his <word> . <unk> <unk>': 1,
    'fibula in his <word> fortuitously , and': 1,
    'fibula in his <word> on 00 november': 1,
    'fibula in his <word> whilst on u00': 1,
    'fire to his <word> on june 00th': 1,
    'first bend the <word> and then the': 1,
    'forced have his <word> amputated after it': 1,
    'forms the lower <word> ; dozer fighter': 1,
    'forms the lower <word> ; yellow vehicle': 1,
    'found in her <word> , which may': 1,
    'fracture in his <word> . <unk> <unk>': 3,
    'fracture of his <word> , ending his': 1,
    'fracture of the <word> above the ankle': 1,
    'fracture of the <word> which would keep': 1,
    'fracture on his <word> which ruled him': 1,
    'fracture to his <word> which protruded several': 1,
    'fractures in her <word> , though she': 1,
    'fucarile lost his <word> and received severe': 1,
    'game with a <word> injury , would': 1,
    'garment that her <word> is crossed and': 1,
    'garter round his <word> ( possibly intended': 1,
    'ground and the <word> bent at the': 1,
    'had his lower <word> amputated in september': 1,
    'had lost his <word> after an injury': 1,
    'had lost his <word> at the battle': 1,
    'half of his <word> amputated . <unk>': 1,
    'half of his <word> in a bomb': 1,
    'half of his <word> in a freak': 1,
    'half of his <word> in an attack': 1,
    'hamstring in her <word> . <unk> <unk>': 1,
    'hand on her <word> . <unk> <unk>': 1,
    'have his lower <word> amputated due to': 1,
    'having lost his <word> in a bombing': 1,
    'he broke his <word> and tore ligaments': 1,
    'he had his <word> amputated . <unk>': 1,
    'he had his <word> amputated above the': 1,
 }

for enwiki_dir in tqdm(config.enwiki_dirs, desc="merging word2contexts"):
    word2contexts_path = os.path.join(
        enwiki_dir, 'word2contexts_w{}_p{}.pkl'.format(
            config.context_window_size,
            int(config.preprocessing)))

    log.info('loading word2context.. {}'.format(word2contexts_path))

    if True:
        cur_word2contexts = cPickle.load(open(word2contexts_path, 'rb'))
    else:
        cur_word2contexts = {}
        for word in answer_dict['dict']:
            cur_word2contexts[word] = dummy

    if word2contexts is None:
        word2contexts = cur_word2contexts
        continue

    def f(words, start, end):
        for word in words[start:end]:
            if word not in word2contexts:
                word2contexts[word] = cur_word2contexts[word]
                continue

            for context, count in cur_word2contexts[word].iteritems():
                if context not in word2contexts[word][context]:
                    word2contexts[word][context] = count
                else:
                    word2contexts[word][context] += count

    words = cur_word2contexts.keys()
    length = len(words)
    num = int(math.ceil(length / num_thread))

    threads = []
    print("start thread")

    for idx in range(num_thread):
        t = Thread(
            target=f,
            args=(words, idx*num, (idx+1)*num))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

#import ipdb; ipdb.set_trace() 
log.info('word2contexts done')

context2word_list = {}
for v in tqdm(word2contexts, desc='build context2word_list'):
    for context in word2contexts[v]:
        if context not in context2word_list:
            context2word_list[context] = []
        context2word_list[context].append(v)

new_context2word_list = {}
for context in tqdm(context2word_list, desc='filter wordset contexts'):
    if len(context2word_list[context]) >= config.min_num_word:
        new_context2word_list[context] = context2word_list[context]

wordlist_with_cnt = []
for context in tqdm(new_context2word_list, desc='wordlist_with_cnt'):
    word_list = new_context2word_list[context]
    count_sum = sum([word2contexts[w][context] for w in word_list])
    wordlist_with_cnt.append((word_list, context, count_sum))

sorted_wordlist_with_cnt_by_t = sorted(
    wordlist_with_cnt,
    key=lambda x: len(x[1].split()) - 1 - x[1].split().count('<unk>'),
    reverse=True)

filtered_sorted_wordlist = [k for k in sorted_wordlist_with_cnt_by_t
                            if '<unk> <word> <unk>' not in k[1]]

reduced_sorted_wordlist = [w for w in filtered_sorted_wordlist
                           if len(w[1].split()) - 1 - w[1].split().count('<unk>') >= 2]
reduced_sorted_wordlist.append((answer_dict['vocab'], '<word>', 1))  # default context
context_list = [w[1] for w in reduced_sorted_wordlist]
context_vocab = set()
for context in context_list:
    for w in context.split():
        context_vocab.add(w)
context_vocab = list(context_vocab)
context_vocab_dict = {w: i for i, w in enumerate(context_vocab)}
context2idx = {context: idx for idx, context in enumerate(context_list)}
context2weight = {context: (len(context.split()) - 1 - context.split().count('<unk>'))**2
                  for context in context_list}
context2weight['<word>'] = 1  # default context
max_context_len = max([len(context.split()) for context in context_list])
np_context = np.zeros([len(context_list), max_context_len], dtype=np.int32)
np_context_len = np.zeros([len(context_list)], dtype=np.int32)
for context in tqdm(context_list, desc='np_context'):
    context_tokens = context.split()
    context_intseq = [context_vocab_dict[t] for t in context_tokens]
    context_intseq_len = len(context_intseq)
    context_idx = context2idx[context]
    np_context[context_idx, :context_intseq_len] = context_intseq
    np_context_len[context_idx] = context_intseq_len

ans2context_idx = {}
for context_tuple in reduced_sorted_wordlist:
    context_idx = context2idx[context_tuple[1]]
    for ans in context_tuple[0]:
        ans_idx = answer_dict['dict'][ans]
        if ans_idx not in ans2context_idx:
            ans2context_idx[ans_idx] = []
        ans2context_idx[ans_idx].append(context_idx)

ans2context_prob = {}
for ans in tqdm(ans2context_idx, desc='ans2context_prob'):
    ans2context_prob[ans] = []
    for context_idx in ans2context_idx[ans]:
        weight = context2weight[context_list[context_idx]]
        ans2context_prob[ans].append(weight)
    partition = float(sum(ans2context_prob[ans]))
    ans2context_prob[ans] = [w / partition for w in ans2context_prob[ans]]

enwiki_context_dict = {
    'idx2context': context_list,
    'context2idx': context2idx,
    'context2weight': context2weight,
    'max_context_len': max_context_len,
    'context_word_vocab': context_vocab,
    'context_word_dict': context_vocab_dict,
    'ans2context_idx': ans2context_idx,
    'ans2context_prob': ans2context_prob,
}
save_name = 'enwiki_context_dict_w{}_p{}_n{}'.format(
    config.context_window_size, config.preprocessing, config.min_num_word)
save_pkl_path = os.path.join(config.dir_name, '{}.pkl'.format(save_name))
log.info('saving: {} ..'.format(save_pkl_path))
cPickle.dump(enwiki_context_dict, open(save_pkl_path, 'wb'))

save_h5_path = os.path.join(config.dir_name, '{}.hdf5'.format(save_name))
log.info('saving: {} ..'.format(save_h5_path))
with h5py.File(save_h5_path, 'w') as f:
    f['np_context'] = np_context
    f['np_context_len'] = np_context_len
log.warn('done')
