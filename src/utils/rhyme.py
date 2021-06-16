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


def critique_poem(poem, language, redif=False, return_pairs=False):
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
        return [0., set()]
    total_score = 0.
    seen_pairs = set()
    rhyme_pairs = set()
    for first in ends:
        max_score = 0.
        pair = (first, first)
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
                pair = (first, second)
            if new_score == 1:
                max_score = 1
                pair = (first, second)
                break
        if max_score > 0.:
            rhyme_pairs.add(pair)
        total_score += max_score
    critique_score = total_score / len(ends)
    return critique_score, rhyme_pairs


def rhyme_tr(first, second):
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
    length = rhyme_length(first, second)
    if length > 1:
        if simplify(first[-1]) in "aeiou":
            for letter in simplify(first)[-length:-1]:
                if letter in "aeiou":
                    return 1
            return 0
        else:
            return 1
    else:
        return 0


def concurrent_score(lines, language, ref, src):
    threads = min(MAX_THREADS, len(lines))
    redif = True if language == "tr" else False
    ref_ends = [set(get_verse_ends(r, redif)) for r in ref]
    src_ends = [set(get_verse_ends(s, redif)) for s in src]

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        scores = list(executor.map(lambda x: critique_poem(x, language, redif), lines))


    rhyme_score = sum(x[0] for x in scores) / len(lines)
    pairs = [x[1] for x in scores]

    pairs_and_ends = zip(pairs, ref_ends, src_ends)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        originalities = list(executor.map(lambda x: score_originality(x[0], x[1], x[2]),
                                     pairs_and_ends))

    copied = sum(x[0] for x in originalities) / len(lines)
    reconstructed = sum(x[1] for x in originalities) / len(lines)
    return rhyme_score, copied, reconstructed


def score_prose_translation(lines, language, src):
    threads = min(MAX_THREADS, len(lines))
    redif = True if language == "tr" else False
    src_ends = [set(get_verse_ends(s, redif)) for s in src]

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        scores = list(executor.map(lambda x: critique_poem(x, language, redif), lines))

    pairs = [x[1] for x in scores]
    rhyme_score = sum(x[0] for x in scores) / len(lines)
    tuples = zip(pairs, src_ends)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        copied = list(executor.map(lambda x: copied_percentage(x[0], x[1]), tuples))

    copied_score = sum(x for x in copied) / len(lines)

    return rhyme_score, copied_score


def score_file(fname, language):
    with open(fname, encoding="utf-8") as infile:
        lines = infile.readlines()
    return concurrent_score(lines, language)


def score_stdin(language):
    import sys
    return concurrent_score(sys.stdin.readlines(), language)


def score_originality(rhyme_pairs, ref_ends, src_ends):
    ref_pairs, src_pairs = set(), set()
    for first in ref_ends:
        for second in ref_ends:
            ref_pairs.add((first, second))
    for first in src_ends:
        for second in src_ends:
            src_pairs.add((first, second))
    all_pairs = len(rhyme_pairs)
    if all_pairs == 0:
        return [0., 0.]
    in_ref, in_src = 0, 0
    for pair in rhyme_pairs:
        a, b = pair
        rev = (b, a)
        if pair in ref_ends or rev in ref_pairs:
            in_ref += 1
        if pair in src_ends or rev in src_pairs:
            in_src += 1
    return [in_src / all_pairs, in_ref / all_pairs]


def copied_percentage(rhyme_pairs, src_ends):
    src_pairs = set()
    for first in src_ends:
        for second in src_ends:
            src_pairs.add((first, second))
    all_pairs = len(rhyme_pairs)
    if all_pairs == 0:
        return 0.
    in_ref, in_src = 0, 0
    for pair in rhyme_pairs:
        a, b = pair
        rev = (b, a)
        if pair in src_ends or rev in src_pairs:
            in_src += 1
    return in_src / all_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str)
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    if args.file is not None:
        print(args.file, score_file(args.file, args.language))
    else:
        print(score_stdin(args.language))