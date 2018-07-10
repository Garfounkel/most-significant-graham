def ngrams(input_list, n):
    return zip(*(input_list[i:] for i in range(n)))


def et1_et2_et3_0(input_text):
    input_list = input_text.lower().split()
    return [ngrams(input_list, 1),
            ngrams(input_list, 2),
            ngrams(input_list, 3)]


input_list = ['all', 'carl', 'more', 'graham', 'or', 'because', 'of']
print('Trigrams:', list(ngrams(input_list, 3)), '\n')

input_text = "Et c'est la reprise de Pavard !"
for i, x in enumerate(et1_et2_et3_0(input_text)):
    print(f'{i+1}:', list(x))
