import sys
import pickle
import os
from collections import defaultdict
from tf_idf import tf_idf, top_n


def print_usage():
    print("usage:")
    print("\t--train outputfile [corpus files, separated by spaces]")
    print("\t--lang inputfolder [corpus files, separated by spaces]")
    print("\t--checklang input-lang.file")
    print("Output file stores one training set's top ngrams while input folder stores all corpuses training results for language classification.")


def train(outputfile, corpus):
    print("Computing most signicant ngrams for", corpus)
    tfidf = tf_idf(corpus)
    top_200 = top_n(tfidf, 200)
    print("Dumping training output...")
    with open(outputfile, 'wb') as file:
        pickle.dump(top_200, file)
    print("Output saved at", outputfile)


def lang_detect(inputfolder, corpus):
    for dirpath, dirnames, filenames in os.walk(inputfolder):
        for inputfile in corpus:
            lang_scores = defaultdict(int)
            ngrams_input = tf_idf([inputfile])
            for filename in filenames:
                path = dirpath + "/" + filename
                # Open languages profiles for comparison:
                with open(path, 'rb') as file:
                    ngrams = pickle.load(file)
                    for ngram, tfidf in ngrams:
                        if ngram in ngrams_input:
                            lang_scores[path] += tfidf * ngrams_input[ngram]
            # Language detection:
            best = (None, 0)
            for lang, score in lang_scores.items():
                print(f'\t{lang}: {score:.10f}')
                if score > best[1]:
                    best = (lang, score)
            print(f'Detected {inputfile} to be from this language: {best[0]}\n')
        break  # Only toplevel dir


def checklang(path):
    with open(path, 'rb') as file:
        ngrams = pickle.load(file)
        print(f"--- Top 200 ngrams:", path)
        try:
            for (ngram, doc), tfidf in ngrams:
                print(f'{ngram}, {doc}:', tfidf)
        except ValueError:
            for ngram, tfidf in ngrams:
                print(f'{ngram}:', tfidf)


if __name__ == '__main__':
    if (len(sys.argv) < 3 or (len(sys.argv) != 3 and sys.argv[1] == '--checklang')):
        print_usage()
        exit(64)

    corpus = sys.argv[3:]

    if (sys.argv[1] == '--train'):
        outputfile = sys.argv[2]
        train(outputfile, corpus)

    if (sys.argv[1] == '--lang'):
        inputfolder = sys.argv[2]
        lang_detect(inputfolder, corpus)

    if (sys.argv[1] == '--checklang'):
        path = sys.argv[2]
        checklang(path)
