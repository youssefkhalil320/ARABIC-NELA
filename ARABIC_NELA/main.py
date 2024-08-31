from modules.arabic_nela import ArabicNELA
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk 


nltk.download('punkt_tab')

# Example usage
arabic_nela = ArabicNELA()
#text = "هذه هي عينة من النص العربي. يمكن أن يحتوي هذا النص على علامات ترقيم، وأحرف كبيرة، وتشكيلات."
text = "شهدت مدينة طرابلس، مساء أمس الأربعاء، احتجاجات شعبية وأعمال شغب لليوم الثالث على التوالي، وذلك بسبب تردي الوضع المعيشي والاقتصادي. واندلعت مواجهات عنيفة وعمليات كر وفر ما بين الجيش اللبناني والمحتجين استمرت لساعات، إثر محاولة فتح الطرقات المقطوعة، ما أدى إلى إصابة العشرات من الطرفين."

# Tokenize the text into sentences and words
sentences = sent_tokenize(text)
words = word_tokenize(text)


puncs_caps_stops = arabic_nela.puncs_caps_stops(text)
ttr_value = arabic_nela.ttr(text)
flesch_kincaid_score = arabic_nela.flesch_kincaid_grade_level(text, words, sentences)
smog_score = arabic_nela.smog_index(text, words, sentences)
coleman_liau_score = arabic_nela.coleman_liau_index(text, words, sentences)
lix_score = arabic_nela.lix(text, words, sentences)
acl_affect_scores = arabic_nela.acl_affect(text)
bias_words_scores = arabic_nela.bias_words(text)


print("puncs_caps_stops: ", puncs_caps_stops)
print("ttr_value: ", ttr_value)
print("flesch_kincaid_score: ", flesch_kincaid_score)
print("smog_score: ", smog_score)
print("coleman_liau_score: ", coleman_liau_score)
print("lix_score: ", lix_score)
print("acl_affect_scores: ", acl_affect_scores)
print("bias_words_scores: ", bias_words_scores)