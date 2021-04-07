#!/usr/bin/env perl
use strict;
use warnings;
use utf8;
binmode STDIN, ':utf8';
binmode STDOUT, ':utf8';

while(<>){
    s/(\S+?)\1{5,}/$1$1$1$1$1/g;
    s/( \S{2,})\1{2,}/$1/g;
    s/( \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;
    s/( \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+ \S+)\1{2,}/$1/g;

    print;
}

