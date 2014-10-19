'''
Baseline.py is a crude algorithm that finds discrepancies between
the topics discussed in two text files (e.g. Wikipedia articles).
The algorithm finds the most frequently occurring nouns in each article,
and computes the symmetric difference.

Sample usage:
baseline.py article_en.txt article_fr.txt

    Description:
    article_en.txt = English Wikipedia article
    article_fr.txt = French Wikipedia article translated into English

'''
import sys
import codecs
import collections
import re

import nltk
from nltk.corpus import stopwords
from nltk.tag.simplify import simplify_wsj_tag

# @param file = relative pathname of an ascii or utf-8 text file
# Returns the contents of the specified text file separated by lines.
def load_file(file):
    f = codecs.open(file, encoding='utf-8') # also works for ascii files
    lines = [line.strip() for line in f.readlines()]
    f.close()

    #remove all symbols
    lines = [(re.sub(r'[^\w ]', '', line)) for line in lines]

    return lines

# @param text = list of strings
# Returns a set of (word, part-of-speech) tuples
def extract_nouns(text):
    tokens = [' '.join(text)][0].split()
    tagged_sent = nltk.pos_tag(tokens)
    simplified = [(word, simplify_wsj_tag(tag)) for word, tag in tagged_sent]
    return simplified

# @param tokens = list of (word, part-of-speech) tuples
# Returns a Counter containing all the nouns that are not stop words.
def extract_noun_tokens(tokens):
    return collections.Counter(token[0] for token in tokens if token[1] == 'N' and
                               not token[0] in stopwords.words('english'))

# @param *_tokens = list of (word, part-of-speech) tuples
# @param num_tokens = extracts this many nouns from each set of tokens (in order of decreasing frequency)
# Returns (list of words underrepresented in english, list of words underrepresented in foreign language).
def compare_most_frequent_tokens(english_tokens, foreign_tokens, num_tokens):
    english_counts = collections.Counter(token[0] for token in english_tokens if token[1] == 'N')
    foreign_counts = collections.Counter(token[0] for token in foreign_tokens if token[1] == 'N')

    underrepresented_in_english = (foreign_counts - english_counts).most_common(num_tokens)
    underrepresented_in_foreign = (english_counts - foreign_counts).most_common(num_tokens)

    return [token[0] for token in underrepresented_in_english], \
           [token[0] for token in underrepresented_in_foreign]

def main():
    args = sys.argv
    if len(args) < 3:
        print 'Usage: baseline.py <english_file> <translated_foreign_file>'
        sys.exit(1)

    # load input files
    english_text = load_file(args[1])
    foreign_text = load_file(args[2])

    # tokenize, strip stop words, and extract nouns (and also other POS?)
    english_tokens = extract_nouns(english_text)
    foreign_tokens = extract_nouns(foreign_text)

    # compare most frequent nouns in each article
    (underrepresented_in_english, underrepresented_in_foreign) = \
        compare_most_frequent_tokens(english_tokens, foreign_tokens, 5)

    print 'Underrepresented in english:', underrepresented_in_english
    print 'Underrepresented in foreign:', underrepresented_in_foreign

if __name__ == '__main__':
    main()