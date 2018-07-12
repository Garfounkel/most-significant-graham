# most-significant-graham
Extracting n-grams for language detection with TF-IDF

## Usage
```
python lang_detect.py --train outputfile [corpus files, separated by spaces]
python lang_detect.py --lang inputfolder [corpus files, separated by spaces]
python lang_detect.py --checklang input-lang
```
Output file stores one training set's top ngrams while input folder stores all corpuses training results for language classification.

### Example:
```
> python lang_detect.py --checklang ../train/lang/danish.out
```
To check which ngrams are the most significant in our danish pre-trained example.

```
> python lang_detect.py --train ../train/lang/fr2.out ../data/fr
```
To create `../train/lang/fr2.out` containing the most significant ngrams from the file `../data/fr`.

```
> python lang_detect.py --train ../train/corpus/c1.out ../data/doc1 ../data/doc2
```
To create `../train/corpus/c1.out` containing the most significant ngrams by TF-IDF from the files `../data/doc1` and `../data/doc2`.

```
> python lang_detect.py --lang ../train/lang ../test/fr.test ../test/da.test ../test/en.test

Expected output:
	../train/lang/en.out: 0.0002471897
	../train/lang/danish.out: 0.0011557960
	../train/lang/fr.out: 0.0087199895
Detected ../test/fr.test to be from this language: ../train/lang/fr.out

	../train/lang/en.out: 0.0002878364
	../train/lang/danish.out: 0.0044952749
Detected ../test/da.test to be from this language: ../train/lang/danish.out

	../train/lang/en.out: 0.0145594223
	../train/lang/danish.out: 0.0000395524
	../train/lang/fr.out: 0.0002480396
Detected ../test/en.test to be from this language: ../train/lang/en.out
```
To detect the language of `../test/fr.test`, `../test/da.test` and `../test/en.test` test files.
