def ngrams(input_list, n=1):
    return zip(*(input_list[i:] for i in range(n)))


def ngrams_streams(input_list, ngram_range=(1, 4)):
    return [ngrams(input_list, x) for x in range(*ngram_range)]


def read_words(filename):
    """Lazy iterator on each words of a file"""
    last = ""
    with open(filename) as file:
        while True:
            buf = file.read(10240).lower()
            if not buf:
                break
            words = (last+buf).split()
            last = words.pop()
            for word in words:
                yield word
        yield last


def ngrams_read_file(ngrams, doc, ngram_range):
    words = read_words(doc)
    buffer = list()
    ngram_count = defaultdict(int)
    while True:
        try:
            for i in range(ngram_range):
                if len(buffer) < i:
                    buffer.append(next(words))
                ngram = buffer[:i]
                ngram_count[i] += 1
                ngram[ngram][doc] += 1
            buffer.pop(0)
        except StopIteration:
            break
    return ngram_count


if __name__ == '__main__':
    input_list = ['all', 'carl', 'more', 'graham', 'or', 'because', 'of']
    print('Trigrams:', list(ngrams(input_list, 3)), '\n')

    input_list = ["Et", "c'est", "la", "reprise", "de", "Pavard!"]
    for i, x in enumerate(ngrams_streams(input_list, ngram_range=(1, 4))):
        print(f'{i+1}:', list(x))
