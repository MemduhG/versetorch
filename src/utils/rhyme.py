import subprocess
from subprocess import PIPE
from TurkishStemmer import TurkishStemmer
import sys
import pronouncing
import string
import argparse
import re
import concurrent.futures

ts = TurkishStemmer()
cache = dict()


def stem_one(word):
    stemmed = ts.stem(word)
    if len(stemmed) < len(word):
        remainder = word[len(stemmed):]
        for symbol in "'′’´ʼ":
            remainder = remainder.strip(symbol)
            stemmed = stemmed.strip(symbol)
        segmented = stemmed + "-" + remainder
    else:
        segmented = stemmed
    text = [segmented]
    return text


def tr_stem(word, bin_loc="/home/memduh/git/TRmorph/segment.fst"):
    if word in cache:
        return cache[word]
    if "3.6.2" in sys.version:
        output = subprocess.run(args=["flookup", bin_loc], input=word.encode("utf-8"), stdout=PIPE)
        text_output = output.stdout.decode("utf-8")
        text = [line.split("\t")[1].rstrip() for line in text_output.rstrip().split("\n")]
    else:
        output = subprocess.run(args=["flookup", bin_loc], input=word, capture_output=True, text=True)
        text = [line.split("\t")[1].rstrip() for line in output.stdout.rstrip().split("\n")]
    text = [x for x in text if x[0] == x[0].lower()]
    if text == ["+?"]:
        text = stem_one(word)
    filtered_voicing = list()
    for item in text:
        stem = item.split("-")[0]
        if len(stem) > len(word):
            continue
        if stem[-1] == word[len(stem) - 1]:
            filtered_voicing.append(item)
    if len(filtered_voicing) != len(text) and len(filtered_voicing) > 0:
        text = filtered_voicing
    for symbol in "'′’´ʼ":
        if symbol not in word:
            text = [x for x in text if symbol not in x]
    if len(text) == 0:
        text = stem_one(word)
    cache[word] = text
    return text


def compare_stems(first, second):
    out = 0.
    if first[-1] == second[-1]:
        out = 0.5
    if first[-2:] == second[-2:]:
        out = 1.
    return out


def rb_stem(first, second):
    first_stemmed, second_stemmed = ts.stem(first), ts.stem(second)
    cutoff_length = min(len(first) - len(first_stemmed), len(second) - len(second_stemmed))
    if cutoff_length > 0 and first[-cutoff_length:] == second[-cutoff_length:]:
        return compare_stems(first[:-cutoff_length], second[:-cutoff_length])
    else:
        return compare_stems(first, second)


MAX_THREADS = 30


alphabet = "0123456789abcdefghijklmnopqrstuvwxyzàáâäæçèéêëíîóôöùúüýćčďęěľłńňřśšťůž+-"
encoder = {alphabet[x]: x for x in range(len(alphabet))}


def rhyme_en(first, second):
    return first in pronouncing.rhymes(second)


def get_verse_ends(poem, redif=True, max_redif=3):
    sep = "¬"
    lines = []
    for line in poem.split(sep):
        lines.append(re.sub("[\s" + string.punctuation + "]*$", " ", line.rstrip()).split())
    ends = []
    for i in range(len(lines)):
        try:
            ends.append(lines[i][-1].translate(str.maketrans('', '', string.punctuation)))
        except IndexError:
            continue
        lines[i] = lines[i][:-1]
    # get identical endings
    if redif:
        ignored_clusters = []
        duplication = True
        added = 0
        while duplication and added < max_redif:
            duplicate_endings = dict()
            for c, end in enumerate(ends):
                if end in duplicate_endings:
                    duplicate_endings[end].append(c)
                else:
                    duplicate_endings[end] = [c]
            duplicate_endings = {x: duplicate_endings[x] for x in duplicate_endings.keys()
                                 if len(duplicate_endings[x]) > 1 and duplicate_endings[x] not in ignored_clusters}
            if len(duplicate_endings) == 0:
                duplication = False
            else:
                extended = False
                for cluster in duplicate_endings:
                    try:
                        for index in duplicate_endings[cluster]:
                            additional_ending = lines[index][-1].translate(
                                str.maketrans('', '', string.punctuation))
                            ends[index] = (additional_ending + " " + ends[index]).strip()
                            lines[index] = lines[index][:-1]
                        extended = True
                    except IndexError:
                        ignored_clusters.append(duplicate_endings[cluster])
                if extended:
                    added += 1

    return ends


def critique_poem(poem, language, redif=False):
    if language == "en":
        rhymefunc = rhyme_en
    elif language == "tr":
        rhymefunc = lambda x, y: rhyme_tr(x, y)
        redif = True
    else:
        rhymefunc = rhyme_cz
    ends = get_verse_ends(poem, redif)
    end_set = set(ends)
    if len(ends) == 0:
        return 0.
    total_score = 0.
    seen_pairs = set()
    for first in ends:
        max_score = 0.
        for second in end_set.difference({first}):
            if (first, second) in seen_pairs or (second, first) in seen_pairs:
                continue
            seen_pairs.add((first, second))
            try:
                new_score = rhymefunc(first, second)
            except (KeyError, IndexError):
                new_score = 0.
            if new_score > max_score:
                max_score = new_score
            if new_score == 1:
                break
        total_score += max_score
    critique_score = total_score / len(ends)
    return critique_score


def rhyme_tr(first, second):
    # TODO: figure out till where the rhyme is.
    if " " in first and " " in second:
        first_list, second_list = first.split(" "), second.split(" ")
        while first_list[-1] == second_list[-1]:
            first_list.pop(-1)
            second_list.pop(-1)
            if len(first_list) == 0 or len(second_list) == 0:
                return 0.
        else:
            return rb_stem(first_list[-1], second_list[-1])
    else:
        return rb_stem(first.split()[-1], second.split()[-1])


simplified = {"á": "a", "é": "e", "ě": "e", "í": "i", "ý": "i", "y": "i", "ň": "n", "ť": "t", "ď": "d",
              "ů": "u", "ó": "o", "m": "n", "ú": "u", "j": "i"}

devoc = {"d": "t", "z": "s", "ž": "š", "b": "p", "ď": "ť", "g": "k", "v": "f"}


def simplify(char):
    if char in simplified:
        return simplified[char]
    else:
        return char


def works(first, second, end=False):
    f, s = simplify(first), simplify(second)
    if f == s:
        return True
    elif end == True:

        if f in devoc:
            if devoc[f] == s:
                return True
        elif s in devoc:
            if devoc[s] == f:
                return True
        else:
            return False
    else:
        return False


def rhyme_length(first, second):
    length = 0
    for a_char, b_char in zip(first[::-1], second[::-1]):
        if length == 0:
            work = works(a_char.lower(), b_char.lower(), True)
        else:
            work = works(a_char.lower(), b_char.lower(), False)
        if work:
            length += 1
        else:
            break

    return length


def rhyme_cz(first, second):
    clean_first, clean_second = first, second
    for sign in "∶‖…":
        clean_first, clean_second = clean_first.replace(sign, " "), clean_second.replace(sign, " ")
    if rhyme_length(first, second) > 1:
        return 1
    elif rhyme_length(first, second) == 1:
        if simplify(first[-1]) in "aeiou":
            return 1
        else:
            return 0
    else:
        return 0


def find_syll_sep(lines):
    candidates = ["-", "/"]
    for line in lines[:min(5, len(lines))]:
        for candidate in candidates:
            if re.search("\w" + candidate + "\w", line) is None:
                candidates.remove(candidate)
    if len(candidates) > 0:
        return candidates[0]
    else:
        return None


def concurrent_score(lines, language, syll_sep=None):
    threads = min(MAX_THREADS, len(lines))
    if syll_sep is None:
        syll_sep = find_syll_sep(lines)
    if syll_sep is not None:
        lines = [re.sub(syll_sep, "", x) for x in lines]
    redif = True if language == "tr" else False

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        scores = executor.map(lambda x: critique_poem(x, language, redif), lines)

    return sum(scores) / len(lines)


def score_file(fname, language):
    with open(fname, encoding="utf-8") as infile:
        lines = infile.readlines()
    return concurrent_score(lines, language)


def score_stdin(language):
    import sys
    return concurrent_score(sys.stdin.readlines(), language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str)
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    if args.file is not None:
        print(args.file, score_file(args.file, args.language))
    else:
        print(score_stdin(args.language))