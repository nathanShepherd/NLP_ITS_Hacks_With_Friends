from textblob import TextBlob
#import nltk
from nltk import word_tokenize
import pandas as pd

data = pd.read_csv("MiFeedback_March 5, 2020_12.00.csv")
res_col = data['Q3'].values
#print(res_col, len(res_col)) # 46 responses

out_df = {'row_idx':[], 'q3_text':[],
		  'sentiment_polarity':[], 
	  'polarity_text':[],
		  'category_by_word_frequency':[],
		  'category_by_sentiment':[]}

# --- --- --- --- --- #

import re
from collections import Counter

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from gensim import models
# --- --- --- --- --- #
new_model = models.Word2Vec.load('gutenburg_100v_3w_model.w2v')

full_text = []
similar_words = []
for i, row in enumerate(res_col):
    clean = re.sub("[^a-zA-Z]", " ", str(row))
    clean = word_tokenize(clean.lower())
    outs = []
    for word in clean:
	if word not in stop_words:
        	outs.append(word)

    for sim_word in new_model.most_similar(outs):
	similar_words.append(sim_word)

    full_text.append(outs)

full_text = full_text[2:] # Removing column headers

# Unpack lists containing each row of words
# Into single list of all words
full_text = [word for sent in full_text for word in sent]
freq_dict = Counter(full_text)
most_freq_words = dict(freq_dict.most_common(10))

most_sim_words = dict(Counter(similar_words).most_common(10))

for i, row in enumerate(res_col):

	blob = TextBlob(row)

	pol = blob.sentiment.polarity
	out_df['sentiment_polarity'].append(pol)

	pol_text = "neutral"
	if pol < -0.1:
		pol_text = 'negative'
	if pol > 0.1:
		pol_text = 'positive'
	out_df['polarity_text'].append(pol_text)
	tokens = re.sub("[^a-zA-Z]", " ", str(row))
	tokens = word_tokenize(tokens.lower())

	found = False
	for word in most_freq_words:
		if word in tokens:
			if found:
				if most_freq_words[word] > most_freq_words[out_df['category_by_word_frequency'][i]]:
					out_df['category_by_word_frequency'][i] = word
			else:
				out_df['category_by_word_frequency'].append(word)
				found = True
	found = False
	for word in most_sim_words:
		if word in tokens:
				if found:
					if most_sim_words[word] > most_sim_words[out_df['category_by_sentiment'][i]]:
						out_df['category_by_sentiment'][i] = word
				else:
					out_df['category_by_sentiment'].append(word)
					found = True
	out_df['q3_text'].append(row)
	out_df['row_idx'].append(i)

out_df = pd.DataFrame(out_df)

data = pd.concat([data, out_df], axis=1)
data.to_csv('Analyzed_Survey_Hacks2020.csv')



