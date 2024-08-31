import re
import string
import math
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from .words_lists import wneg_list, wpos_list, wneu_list, bias_words_list, assertatives_list, factives_list, hedges_list, implicatives_list, report_verbs_list


nltk.download('stopwords')
nltk.download('punkt')  # Ensure that tokenizers are available
nltk.download('punkt_tab')



class ArabicNELA(object):
    '''
    Functions for individual features or to support feature groups in Arabic text.
    '''

    def __init__(self):
        # Initialize components for Arabic text processing
        self.arabic_punctuation = set("،؛؟!«»")
        self.arabic_stopwords = set(stopwords.words('arabic'))
        self.stemmer = ISRIStemmer()

    # Helper Functions
    def _normalize_word(self, word):
        return word.strip().lower()

    def remove_diacritics(self, text):
        # Removing Arabic diacritics
        arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        return re.sub(arabic_diacritics, '', text)

    def get_filtered_words(self, text):
        # Tokenize the text into words
        tokens = word_tokenize(text)
        filtered_words = []
        for tok in tokens:
            tok = self.remove_diacritics(tok)
            if tok in self.arabic_punctuation or tok == " ":
                continue
            else:
                new_word = "".join([c for c in tok if c not in string.punctuation])
                if new_word == "" or new_word == " ":
                    continue
                filtered_words.append(new_word)
        return filtered_words

    # Style Functions
    def LIWC(self, text):
        # Process text for LIWC analysis
        tokens = self.get_filtered_words(text)
        counts_dict = defaultdict(int)
        stemmed_tokens = [self.stemmer.stem(t) for t in tokens]
        # Assuming you have a dictionary similar to LIWC for Arabic:
        for stem in ARABIC_LIWC_STEM_DICT:
            count = sum(1 for token in stemmed_tokens if token.startswith(stem.replace("*", "")))
            if count > 0:
                for cat in ARABIC_LIWC_STEM_DICT[stem]:
                    counts_dict[cat] += count
        counts_dict_norm_with_catnames = {ARABIC_LIWC_CAT_DICT[k]: float(c) / len(tokens) for k, c in counts_dict.items()}
        return counts_dict_norm_with_catnames

    def POS_counts(self, text): #TODO: this is not gonna work for Arabic 
        # Process text for POS tagging
        tokens = word_tokenize(text)
        pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
                    "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "WP$", "WRB", "VB", "VBD", "VBG",
                    "VBN", "VBP", "VBZ", "WDT", "WP", "$", "''", "(", ")", ",", "--", ".", ":", "``"]
        tag_to_count = {t: 0 for t in pos_tags}
        tagged_words = pos_tag(tokens)  # You need an Arabic POS tagger for this
        for word, tag in tagged_words:
            if tag in tag_to_count:
                tag_to_count[tag] += 1
        tag_to_count_norm = {t: float(n) / len(tokens) for t, n in tag_to_count.items()}
        return tag_to_count_norm

    def puncs_caps_stops(self, text):
        # Process text for punctuation, capitalization, and stopwords
        tokens = word_tokenize(text)
        quotes = float((tokens.count("\"") + tokens.count('«') + tokens.count('»'))) / len(tokens)
        exclaim = float(tokens.count("!")) / len(tokens)
        allpunc = 0
        for p in self.arabic_punctuation:
            allpunc += tokens.count(p)
        allpunc = float(allpunc) / len(tokens)
        words_upper = sum([1 for w in tokens if w.isupper()])
        stops = float(len([s for s in tokens if s in self.arabic_stopwords])) / len(tokens)

        result = {
            "quotes": quotes,
            "exclaim": exclaim,
            "allpunc": allpunc,
            "stops": stops
        }

        return result

    # Complexity Functions
    def ttr(self, text):
        # Calculate the Type-Token Ratio (TTR) for the text
        tokens = word_tokenize(text)
        dif_words = len(set(tokens))
        tot_words = len(tokens)
        ttr = float(dif_words) / tot_words
        return ttr

    def count_syllables(self, word):
        word = self._normalize_word(word)
        if not word:
            return 0
        word = self.remove_diacritics(word)
        count = 0
        vowel_pattern = re.compile(r'[اوي]')
        syllable_pattern = re.compile(r'[اوي]+')
        
        for syllable in syllable_pattern.findall(word):
            if syllable:
                count += 1
        
        return count

    def count_complex_words(self, tokens, sentences):
        words = tokens
        complex_words = 0
        found = False
        cur_word = []
        for word in words:
            if self.count_syllables(word) >= 3:
                if not(word[0].isupper()):  # Arabic doesn't have capitalization, so this might need adjustment
                    complex_words += 1
                else:
                    for sentence in sentences:
                        if word in sentence:
                            found = True
                            break
                    if found:
                        complex_words += 1
                        found = False
        return complex_words

    def flesch_kincaid_grade_level(self, text, words, sentences):
        score = 0.0
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_p_sentence = word_count / sentence_count if sentence_count > 0 else 0
        syllable_count = 0
        for word in words:
            syllable_count += self.count_syllables(word)
        if word_count > 0:
            score = 0.39 * (avg_words_p_sentence) + 11.8 * (syllable_count / word_count) - 15.59
        rounded_score = round(score, 4)
        return rounded_score

    def smog_index(self, text, words, sentences):
        score = 0.0
        word_count = len(words)
        sentence_count = len(sentences)
        complex_word_count = self.count_complex_words(words, sentences)
        if sentence_count > 0:
            score = (math.sqrt(complex_word_count * (30 / sentence_count)) + 3)
        return score

    def coleman_liau_index(self, text, words, sentences):
        score = 0.0
        word_count = len(words)
        sentence_count = len(sentences)
        characters = 0
        for word in words:
            characters += len(word)
        if word_count > 0:
            score = (5.89 * (characters / word_count)) - (30 * (sentence_count / word_count)) - 15.8
        rounded_score = round(score, 4)
        return rounded_score

    def lix(self, text, words, sentences):
        longwords = 0.0
        score = 0.0
        word_count = len(words)
        sentence_count = len(sentences)
        if word_count > 0:
            for word in words:
                if len(word) >= 7:
                    longwords += 1.0
            score = (word_count / sentence_count) + (100 * longwords / word_count)
        return score

    def acl_affect(self, words):
        # Calculate sentiment proportions in Arabic
        wneg_count = float(sum([words.count(n) for n in wneg_list])) / len(words)
        wpos_count = float(sum([words.count(n) for n in wpos_list])) / len(words)
        wneu_count = float(sum([words.count(n) for n in wneu_list])) / len(words)

        result = {
            "wneg_count": wneg_count,
            "wpos_count": wpos_count,
            "wneu_count": wneu_count
        }

        return result

    def bias_words(self, words):
        bigrams = [" ".join(bg) for bg in ngrams(words, 2)]
        trigrams = [" ".join(tg) for tg in ngrams(words, 3)]
        bias = float(sum([words.count(b) for b in bias_words_list])) / len(words)
        assertatives = float(sum([words.count(a) for a in assertatives_list])) / len(words)
        factives = float(sum([words.count(f) for f in factives_list])) / len(words)
        hedges = sum([words.count(h) for h in hedges_list]) + \
            sum([bigrams.count(h) for h in hedges_list]) + \
            sum([trigrams.count(h) for h in hedges_list])
        hedges = float(hedges) / len(words)
        implicatives = float(sum([words.count(i) for i in implicatives_list])) / len(words)
        report_verbs = float(sum([words.count(r) for r in report_verbs_list])) / len(words)
        positive_op = float(sum([words.count(p) for p in wpos_list])) / len(words)
        negative_op = float(sum([words.count(n) for n in wneg_list])) / len(words)

        result = {
            "bias": bias,
            "assertatives": assertatives,
            "factives": factives,
            "hedges": hedges,
            "implicatives": implicatives,
            "report_verbs": report_verbs,
            "positive_op": positive_op,
            "negative_op": negative_op
        }

        return result