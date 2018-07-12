from collections import defaultdict
from ngrams import ngrams_read_file
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
    """Return a list containing the top n ngrams by tfidf"""
    return [(k, tfidf[k]) for k in sorted(tfidf, key=tfidf.get, reverse=True)][:n]


if __name__ == '__main__':
    # print_dict_of_dicts(corpus_stats(["text.txt", "text0.txt"])[0])
    # print()

    import sys
    import pickle
    import os


    if (len(sys.argv) < 4):
        print("usage:")
        print("\t--train outputfile [corpus files, separated by spaces]")
        print("\t--lang inputfolder [corpus files, separated by spaces]")
        print("Output file stores one training set's top ngrams while input folder stores all corpuses training results for language classification.")
        exit(2)

    corpus = sys.argv[3:]

    if (sys.argv[1] == '--train'):
        outputfile = sys.argv[2]
        tfidf = tf_idf(corpus)
        top_200 = top_n(tfidf, 200)
        with open(outputfile, 'wb') as file:
            pickle.dump(top_200, file)

    if (sys.argv[1] == '--lang'):
        inputfolder = sys.argv[2]
        for dirpath, dirnames, filenames in os.walk(inputfolder):
            ngrams_data = {}
            for filename in filenames:
                path = dirpath + "/" + filename
                with open(path, 'rb') as file:
                    ngrams = pickle.load(file)
                    ngrams_data[path] = ngrams
                    print(f"--- Top 200 ngrams:", path)
                    for (ngram, doc), tfidf in ngrams:
                        print(f'{ngram}, {doc}:', tfidf)
            break  # Only toplevel dir
