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


if __name__ == '__main__':
    import sys
    import pickle
    import os


    if (len(sys.argv) < 3 or (len(sys.argv) != 3 and sys.argv[1] == '--checklang')):
        print("usage:")
        print("\t--train outputfile [corpus files, separated by spaces]")
        print("\t--lang inputfolder [corpus files, separated by spaces]")
        print("\t--checklang input-lang.file")
        print("Output file stores one training set's top ngrams while input folder stores all corpuses training results for language classification.")
        exit(2)

    corpus = sys.argv[3:]

    if (sys.argv[1] == '--train'):
        outputfile = sys.argv[2]
        print("Computing most signicant ngrams for", corpus)
        tfidf = tf_idf(corpus)
        top_200 = top_n(tfidf, 200)
        print("Dumping training output...")
        with open(outputfile, 'wb') as file:
            pickle.dump(top_200, file)
        print("Output saved at", outputfile)

    if (sys.argv[1] == '--lang'):
        inputfolder = sys.argv[2]
        for dirpath, dirnames, filenames in os.walk(inputfolder):
            for inputfile in corpus:
                lang_scores = defaultdict(int)
                ngrams_input = tf_idf([inputfile])
                for filename in filenames:
                    path = dirpath + "/" + filename
                    with open(path, 'rb') as file:
                        ngrams = pickle.load(file)
                        for ngram, tfidf in ngrams:
                            if ngram in ngrams_input:
                                lang_scores[path] += tfidf * ngrams_input[ngram]
                # Language detection:
                best = (None, 0)
                for lang, score in lang_scores.items():
                    print(f'{lang}: {score}')
                    if score > best[1]:
                        best = (lang, score)
                lang = best[0]
                print(f'Detected {inputfile} to be from this language: {lang}')
            break  # Only toplevel dir

    if (sys.argv[1] == '--checklang'):
        path = sys.argv[2]
        with open(path, 'rb') as file:
            ngrams = pickle.load(file)
            print(f"--- Top 200 ngrams:", path)
            try:
                for (ngram, doc), tfidf in ngrams:
                    print(f'{ngram}, {doc}:', tfidf)
            except ValueError:
                for ngram, tfidf in ngrams:
                    print(f'{ngram}:', tfidf)
