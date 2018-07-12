from collections import defaultdict
from ngrams import ngrams_read_file
import math


def corpus_stats(corpus, ngram_range=(1, 4)):
    """corpus_stats(corpus) --> map(ngram: map(doc_n: count)), map(doc_n: count)

    Arguments:
    corpus -- The list of path to each documents text file
    """
    ngrams = defaultdict(lambda : defaultdict(int))
    words_per_doc = defaultdict(dict)
    for doc in corpus:
        # ngram count per documents (stored in ngrams):
        ngram_number = ngrams_read_file(ngrams, doc, ngram_range)
        # Wordcount per document:
        words_per_doc[doc] = ngram_number
    return ngrams, words_per_doc


def tf_idf(corpus):
    """tf_idf(corpus) --> map(ngram: map(doc_n: tf-idf))

    Arguments:
    corpus -- The list of path to each documents text file
    """
    result = dict()
    ngram_count, words_per_doc = corpus_stats(corpus)
    for ngram, docs in ngram_count.items():
        # Computing idf:
        nb_docs_ngram_is_in = len(docs)
        nb_docs_in_corpus = len(corpus)
        idf = math.log(nb_docs_in_corpus / nb_docs_ngram_is_in)

        # Computing tf-idf:
        for doc, count in docs.items():
            tf = count / words_per_doc[doc][len(ngram)]
            if (len(corpus) == 1):
                tfidf = tf
                result[ngram] = tfidf
            else:
                tfidf = tf * idf
                result[(ngram, doc)] = tfidf
    return result


def top_n(tfidf, n):
    """Return a list containing the top n ngrams by tfidf"""
    return [(k, tfidf[k]) for k in sorted(tfidf, key=tfidf.get, reverse=True)][:n]
