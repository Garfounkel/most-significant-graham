from collections import defaultdict
from ngrams import ngrams_streams
import math


def flatten(*iterables):
    """flatten((A, B, C), (D, E, F)) --> A B C D E F """
    for it in iterables:
        for elements in it:
            for element in elements:
                yield element


def corpus_stats(corpus, ngram_range=(1, 4)):
    """corpus_stats(corpus) --> map(ngram: map(doc_n: count)), map(doc_n: count)

    Arguments:
    corpus -- The list of path to each documents text file
    """
    ngram_count = defaultdict(lambda : defaultdict(int))
    words_per_doc = dict()
    for doc in corpus:
        with open(doc) as text:
            input_text = text.read()
            input_list = input_text.lower().split()

            # Wordcount per document:
            words_per_doc[doc] = len(input_list)

            # ngram count per documents:
            ngrams_1_to_3 = ngrams_streams(input_list, ngram_range)
            for ngram in flatten(ngrams_1_to_3):
                ngram_count[ngram][doc] += 1
    return ngram_count, words_per_doc


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
            tf = count / words_per_doc[doc]
            tfidf = tf * idf
            result[(ngram, doc)] = tfidf
    return result


def print_dict_of_dicts(dict_of_dicts):
    """Pretty-print dict of dicts"""
    for key1, value1 in dict_of_dicts.items():
        print(f'{key1}: {{', end='')
        for key2, value2 in value1.items():
            print(f'{key2}: {value2}, ', end='')
        print("}")


def top_n(tfidf, n):
    return [(k, tfidf[k]) for k in sorted(tfidf, key=tfidf.get, reverse=True)][:n]


if __name__ == '__main__':
    # print_dict_of_dicts(corpus_stats(["text.txt", "text0.txt"])[0])
    # print()

    n = 10
    print(f"--- Top {n} ngrams:")
    tfidf = tf_idf(["text.txt", "text0.txt"])
    for (ngram, doc), tfidf in top_n(tfidf, n):
        print(f'{ngram}, {doc}:', tfidf)
