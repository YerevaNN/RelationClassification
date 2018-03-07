import random
import utils
import gensim
import json

filename = 'data/all_data_no_amr(1).json'


pubmed_47_papers = [
    'eval-auto-amr_a_pmid_1003_7796',
    'eval-auto-amr_a_pmid_1003_7797',
    'eval-auto-amr_a_pmid_1008_5302',
    'eval-auto-amr_a_pmid_1019_0900',
    'eval-auto-amr_a_pmid_1020_9034',
    'eval-auto-amr_a_pmid_1020_9036',
    'eval-auto-amr_a_pmid_1022_4278',
    'eval-auto-amr_a_pmid_1037_7187',
    'eval-auto-amr_a_pmid_1040_2461',
    'eval-auto-amr_a_pmid_1045_9009',
    'eval-auto-amr_a_pmid_1047_7754',
    'eval-auto-amr_a_pmid_1068_2683',
    'eval-auto-amr_a_pmid_1076_9033',
    'eval-auto-amr_a_pmid_1130_6741',
    'eval-auto-amr_a_pmid_1133_5189',
    'eval-auto-amr_a_pmid_1167_5267',
    'eval-auto-amr_a_pmid_1267_6607',
    'eval-auto-amr_a_pmid_1455_7412',
    'eval-auto-amr_a_pmid_1555_0174',
    'eval-auto-amr_a_pmid_1580_4364',
    'eval-auto-amr_a_pmid_1618_7797',
    'eval-auto-amr_a_pmid_1625_5777',
    'eval-auto-amr_a_pmid_1638_0507',
    'eval-auto-amr_a_pmid_1692_3831',
    'eval-auto-amr_a_pmid_1760_9666',
    'eval-auto-amr_a_pmid_1862_9356',
    'eval-auto-amr_a_pmid_1909_5655',
    'eval-auto-amr_a_pmid_1947_3551',
    'eval-auto-amr_a_pmid_1967_2408',
    'eval-auto-amr_a_pmid_1984_9856',
    'eval-auto-amr_a_pmid_2012_6316',
    'eval-auto-amr_a_pmid_2013_7092',
    'eval-auto-amr_a_pmid_2019_9942',
    'eval-auto-amr_a_pmid_2059_8115',
    'eval-auto-amr_a_pmid_2080_4622',
    'eval-auto-amr_a_pmid_2095_7165',
    'eval-auto-amr_a_pmid_2181_9631',
    'eval-auto-amr_a_pmid_2224_6136',
    'eval-auto-amr_a_pmid_2247_9552',
    'eval-auto-amr_a_pmid_2260_7622',
    'eval-auto-amr_a_pmid_2325_1655',
    'eval-auto-amr_a_pmid_2340_5234',
    'eval-auto-amr_a_pmid_2363_7816',
    'eval-auto-amr_a_pmid_2400_8832',
    'eval-auto-amr_a_pmid_2436_7698',
    'eval-auto-amr_a_pmid_2484_7881',
    'eval-auto-amr_a_pmid_749_8097'
]

pubmed_47_papers_test = [
    'eval-auto-amr_a_pmid_1003_7797',
    'eval-auto-amr_a_pmid_1008_5302',
    'eval-auto-amr_a_pmid_1019_0900',
    'eval-auto-amr_a_pmid_1020_9034',
    'eval-auto-amr_a_pmid_1020_9036',
    'eval-auto-amr_a_pmid_1022_4278',
    'eval-auto-amr_a_pmid_1037_7187',
    'eval-auto-amr_a_pmid_1040_2461',
]


with open(filename) as fd:
    data = json.load(fd).items()

for sample in data:
    if len(sample[1]['interaction_tuple']) == 3:
        sample[1]['interaction_tuple'].insert(1, None)

bionlp_train_ids = [sample[0] for sample in data if 'bionlp' in sample[0].lower() and 
                                                    'devel' not in sample[0].lower()]

random.seed(123)
bionlp_papers = list(set([x.split('/')[-2] for x in bionlp_train_ids]))
random.shuffle(bionlp_papers)
bionlp_valid_papers = set(bionlp_papers[:300])

random.seed(123)
pubmed_47_papers_non_test = [x for x in pubmed_47_papers if x not in pubmed_47_papers_test]
random.shuffle(pubmed_47_papers_non_test)
pubmed_47_papers_valid = pubmed_47_papers_non_test[:13]
pubmed_47_papers_train = pubmed_47_papers_non_test[13:]

bionlp_test = [sample[1] for sample in data if 'bionlp' in sample[0].lower() and 
                                                'devel' in sample[0].lower()]

bionlp_train = [sample[1] for sample in data if 'bionlp' in sample[0].lower() and 
                                               'devel' not in sample[0].lower() and 
                                               sample[0].split('/')[-2] not in bionlp_valid_papers]

bionlp_valid = [sample[1] for sample in data if 'bionlp' in sample[0].lower() and 
                                               'devel' not in sample[0].lower() and 
                                               sample[0].split('/')[-2] in bionlp_valid_papers]

chicago_train = [sample[1] for sample in data if 'bionlp' not in sample[0].lower() and 
                                                'chicago' in sample[0].lower()]
random.seed(123)

chicago_train_5000 = [sample for sample in chicago_train if (sample['label'] == 0) or
                                                            (sample['label'] == 1 and random.random() < 5. / 23.)]

pubmed_47_train = [sample[1] for sample in data if 'bionlp' not in sample[0].lower() and 
                                                   'chicago' not in sample[0].lower() and
                                                   sum([1 for pubmed_47_paper in pubmed_47_papers_train if pubmed_47_paper.lower() in sample[0].lower()]) > 0]

pubmed_47_valid = [sample[1] for sample in data if 'bionlp' not in sample[0].lower() and 
                                                   'chicago' not in sample[0].lower() and
                                                   sum([1 for pubmed_47_paper in pubmed_47_papers_valid if pubmed_47_paper.lower() in sample[0].lower()]) > 0]

pubmed_47_test = [sample[1] for sample in data if 'bionlp' not in sample[0].lower() and 
                                                   'chicago' not in sample[0].lower() and
                                                   sum([1 for pubmed_47_paper in pubmed_47_papers_test if pubmed_47_paper.lower() in sample[0].lower()]) > 0]

pubmed_other_train = [sample[1] for sample in data if 'bionlp' not in sample[0].lower() and 
                                                      'chicago' not in sample[0].lower() and
                                                      sum([1 for pubmed_47_paper in pubmed_47_papers_train if pubmed_47_paper.lower() in sample[0].lower()]) == 0 and
                                                      sum([1 for pubmed_47_paper in pubmed_47_papers_valid if pubmed_47_paper.lower() in sample[0].lower()]) == 0 and
                                                      sum([1 for pubmed_47_paper in pubmed_47_papers_test if pubmed_47_paper.lower() in sample[0].lower()]) == 0]


                        
# def split_train_val(data):
#     train_sentences = set([])
#     valid_sentences = set([])

#     train, valid = [], []
#     for sample in data:
#         if sample[u'text'] in train_sentences:
#             train.append(sample)
#         elif sample[u'text'] in valid_sentences:
#             valid.append(sample)
#         else:
#             if random.uniform(0,1) > 0.2:
#                 train_sentences.add(sample[u'text'])
#                 train.append(sample)
#             else:
#                 valid_sentences.add(sample[u'text'])
#                 valid.append(sample)

#     return train, valid

# bionlp_train, bionlp_valid = split_train_val(bionlp_data)
# chicago_train, chicago_valid = split_train_val(chicago_data)
# remaining_train, remaining_valid = split_train_val(remaining_data)

print('Saving bionlp_train')
with open('data/bionlp_train_data.json', 'w') as fd:
    json.dump(bionlp_train, fd)

print('Saving bionlp_valid')
with open('data/bionlp_valid_data.json', 'w') as fd:
    json.dump(bionlp_valid, fd)

print('Saving bionlp_test')
with open('data/bionlp_test_data.json', 'w') as fd:
    json.dump(bionlp_test, fd)


# w2v_model = gensim.models.KeyedVectors.load_word2vec_format('pubmed.bin', binary=True, unicode_errors='ignore')
#
#
# all_data_train_X, all_data_train_Y = utils.process_data_attn(bionlp_train + pubmed_47_train + pubmed_other_train, w2v_model)
# all_data_valid_X, all_data_valid_Y = utils.process_data_attn(bionlp_valid + pubmed_47_valid, w2v_model)
# all_data_test_X, all_data_test_Y = utils.process_data_attn(bionlp_test + pubmed_47_test, w2v_model)
#
# bionlp_train_X, bionlp_train_Y = utils.process_data_attn(bionlp_train, w2v_model)
# bionlp_valid_X, bionlp_valid_Y = utils.process_data_attn(bionlp_valid, w2v_model)
# bionlp_test_X, bionlp_test_Y = utils.process_data_attn(bionlp_test, w2v_model)
# chicago_train_X, chicago_train_Y = utils.process_data_attn(chicago_train, w2v_model)
# chicago_train_5000_X, chicago_train_5000_Y = utils.process_data_attn(chicago_train_5000, w2v_model)
# pubmed_47_train_X, pubmed_47_train_Y = utils.process_data_attn(pubmed_47_train, w2v_model)
# pubmed_47_valid_X, pubmed_47_valid_Y = utils.process_data_attn(pubmed_47_valid, w2v_model)
# pubmed_47_test_X, pubmed_47_test_Y = utils.process_data_attn(pubmed_47_test, w2v_model)
# pubmed_other_train_X, pubmed_other_train_Y = utils.process_data_attn(pubmed_other_train, w2v_model)

# print('Saving bionlp_train')
# with open('data/not_normed/bionlp_train_data.json', 'w') as fd:
#     json.dump((bionlp_train_X, bionlp_train_Y), fd)
#
# print('Saving bionlp_valid')
# with open('data/not_normed/bionlp_valid_data.json','w') as fd:
#     json.dump((bionlp_valid_X, bionlp_valid_Y), fd)
#
# print('Saving bionlp_test')
# with open('data/not_normed/bionlp_test_data.json','w') as fd:
#     json.dump((bionlp_test_X, bionlp_test_Y), fd)
#
# print('Saving chicago_train')
# with open('data/not_normed/chicago_train_data.pkl', 'w') as fd:
#     pickle.dump((chicago_train_X, chicago_train_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
#
# print('Saving chicago_train_5000')
# with open('data/not_normed/chicago_train_data_5000.pkl', 'w') as fd:
#     pickle.dump((chicago_train_5000_X, chicago_train_5000_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
#
# print('Saving pubmed_47_train')
# with open('data/not_normed/pubmed_47_train_data.pkl', 'w') as fd:
#     pickle.dump((pubmed_47_train_X, pubmed_47_train_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
#
# print('Saving pubmed_47_test')
# with open('data/not_normed/pubmed_47_test_data.pkl', 'w') as fd:
#     pickle.dump((pubmed_47_test_X, pubmed_47_test_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
#
# print('Saving pubmed_47_valid')
# with open('data/not_normed/pubmed_47_valid_data.pkl', 'w') as fd:
#     pickle.dump((pubmed_47_valid_X, pubmed_47_valid_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
#
# print('Saving pubmed_other')
# with open('data/not_normed/pubmed_other_train_data.pkl','w') as fd:
#     pickle.dump((pubmed_other_train_X, pubmed_other_train_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
#
# print('Saving train')
# with open('data/not_normed/train_data.pkl', 'w') as fd:
#     pickle.dump((all_data_train_X, all_data_train_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
#
# print('Saving valid')
# with open('data/not_normed/valid_data.pkl','w') as fd:
#     pickle.dump((all_data_valid_X, all_data_valid_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
#
# print('Saving test')
# with open('data/not_normed/test_data.pkl','w') as fd:
#     pickle.dump((all_data_test_X,  all_data_test_Y), fd, protocol = pickle.HIGHEST_PROTOCOL)
