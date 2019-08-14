from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
import torch
from tqdm import tqdm
import glob
from flair.data import Dictionary
from segtok import tokenizer
import collections


class Flair_Rap:

    @classmethod
    def learn_char_model(cls,dict_file = 'dicts/char_dict',is_forward_lm = True, hidden_size=128, nlayers=2, sequence_length=15, mini_batch_size=16, max_epochs =20):

        dictionary = Dictionary.load_from_file(dict_file)
        corpus = TextCorpus('corpus', dictionary, is_forward_lm, character_level=True)
        language_model = LanguageModel(dictionary, is_forward_lm, hidden_size=hidden_size, nlayers=nlayers)
        trainer = LanguageModelTrainer(language_model, corpus)
        trainer.train('traning/char_model', sequence_length=sequence_length, mini_batch_size=mini_batch_size, max_epochs=max_epochs)

    @classmethod
    def learn_word_model(cls,dict_file = 'dicts/word_dict',is_forward_lm = True, hidden_size=128, nlayers=2, sequence_length=10, mini_batch_size=16, max_epochs =20):

        dictionary = Dictionary.load_from_file(dict_file)
        corpus = TextCorpus('corpus', dictionary, is_forward_lm, character_level=False)
        language_model = LanguageModel(dictionary, is_forward_lm, hidden_size, nlayers)
        trainer = LanguageModelTrainer(language_model, corpus)
        trainer.train('traning/word_model', sequence_length, mini_batch_size, max_epochs)

    @classmethod
    def predict_sequence(cls,dataset = 'models/word-lm.pt', seq_length = 100, mode='word'):

        torch.device('cpu')
        print('Loadning model')
        model = LanguageModel.load_language_model(dataset)
        print('Model loaded')
        idx2item = model.dictionary.idx2item
        hidden = model.init_hidden(1)
        input = torch.rand(1, 1).mul(len(idx2item)).long().cuda()
        sequence = []
        print('Generating sequence')
        for _ in tqdm(range(seq_length)):
            prediction, rnn_output, hidden = model.forward(input, hidden)
            word_weights = prediction.squeeze().data.div(1.0).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)
            word = idx2item[word_idx].decode('UTF-8')
            if (word != '<unk>'):
                sequence.append(word)
        if(mode == 'word'):
            verse = ' '.join(sequence)
        if(mode == 'char'):
            verse = ''.join(sequence)

        return (verse)
    
    @classmethod
    def create_word_dict(cls, files = ['corpus/full_text.txt'], filename= 'dicts/word_dict'):

        word_dictionary: Dictionary = Dictionary()
        counter = collections.Counter()
        processed = 0
        for file in files:
            print(file)
            with open(file, 'r', encoding='utf-8') as f:
                tokens = 0
                for line in f:
                    processed += 1
                    chars = list(tokenizer.word_tokenizer(line))
                    tokens += len(chars)

                    counter.update(chars)
        total_count = 0
        for letter, count in counter.most_common():
            total_count += count
        print(total_count)
        print(processed)
        sum = 0
        idx = 0
        for letter, count in counter.most_common():
            sum += count
            percentile = (sum / total_count)
            word_dictionary.add_item(letter)
            idx += 1
            print('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))

        print(word_dictionary.item2idx)

        import pickle
        with open(filename, 'wb+') as f:
            mappings = {
                'idx2item': word_dictionary.idx2item,
                'item2idx': word_dictionary.item2idx
            }
            pickle.dump(mappings, f)
            print("done")

    @classmethod
    def create_char_dict(cls, files=['corpus/full_text.txt'], filename= 'dicts/char_dict'):

        char_dictionary: Dictionary = Dictionary()
        counter = collections.Counter()
        processed = 0
        for file in files:
            print(file)
            with open(file, 'r', encoding='utf-8') as f:
                tokens = 0
                for line in f:
                    processed += 1
                    chars = list(line)
                    tokens += len(chars)

                    counter.update(chars)
        total_count = 0
        for letter, count in counter.most_common():
            total_count += count
        print(total_count)
        print(processed)
        sum = 0
        idx = 0
        for letter, count in counter.most_common():
            sum += count
            percentile = (sum / total_count)
            char_dictionary.add_item(letter)
            idx += 1
            print('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))

        print(char_dictionary.item2idx)

        import pickle
        with open(filename, 'wb+') as f:
            mappings = {
                'idx2item': char_dictionary.idx2item,
                'item2idx': char_dictionary.item2idx
            }
            pickle.dump(mappings, f)
            print("done")