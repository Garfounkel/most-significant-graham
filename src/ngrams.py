def ngrams(input_list, n=1):
    return zip(*(input_list[i:] for i in range(n)))


def ngrams_streams(input_list, ngram_range=(1, 4)):
    return [ngrams(input_list, x) for x in range(*ngram_range)]


if __name__ == '__main__':
    input_list = ['all', 'carl', 'more', 'graham', 'or', 'because', 'of']
    print('Trigrams:', list(ngrams(input_list, 3)), '\n')

    input_list = ["Et", "c'est", "la", "reprise", "de", "Pavard!"]
    for i, x in enumerate(ngrams_streams(input_list, ngram_range=(1, 4))):
        print(f'{i+1}:', list(x))
