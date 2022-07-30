# NMT系统分析

## Intro

CS224N-22WINTER-A4-WRITTEN

## a

Cherokee语是多词素综合语言，一个单词可能含有一个正常语句的意思，拆分成字词级别建模，实际上类似于采用word embeddings进行建模，用词这一粒度建模难度较大

## b

由于Cherokee语言的特点，一个字符（子词）可能包含多个词素，与上述原因相同，在音译的文本上建模可能可以进一步拆分词素，降低建模难度，对词素显式建模

## c

迁移学习，paper-to-read

## d1

problem1(p1)：代词一致性缺失，解决该问题可以在decoder加attention

## f

### f1

$$
p_1 = \frac{12}{13} \\
p_2 = \frac{10}{12} \\
len_{c_1} = 13, len_r = 13, BP = 1\\
BLEU_{c_1} = e^{\frac{1}{2}\log p_1 + \frac{1}{2}\log p_2} = e^{\frac{1}{2}\log p_1p_2} = p_1p_2^\frac{1}{2} (取e为底数)\\
= \frac{10}{13}^\frac{1}{2}
$$

$$
p_1 = \frac{12}{13} \\
p_2 = \frac{9}{12} \\
len_{c_2} = 13, len_r = 13, BP = 1\\
BLEU_{c_2} = p_1p_2^{\frac{1}{2}} = \frac{9}{13} ^ {\frac{1}{2}}\\
BLEU_{c_1} \gt BLEU_{c_2}
$$



可以看到c1翻译bleu值更高，实际翻译质量也是如此

### f2

参考例句只留下r1，仿照上述计算过程可以算得：

$$
BLEU_{c_1} = \frac{80}{12 * 13} ^ \frac{1}{2},
BLEU_{c_2} = \frac{108}{12 * 13} ^ \frac{1}{2} \\
BLEU_{c_1} \lt BLEU_{c_2}
$$

此时c2翻译的bleu值更高，实际翻译质量不然

### f3

![image-20220730114955844](/lab/a4/written-part/NMT系统分析.assets/image-20220730114955844.png)

若参考例句不够，例如2中的情况，一些流畅、高质量翻译，可能因为与Reference的重叠度不高而降低bleu指标，相反例如c2，候选翻译c2是不流畅的，但只是因为与r1有较多反复与重叠使bleu值升高。reference数目太少会导致合理的翻译没有相应的参考而降低bleu值，另一方面不流畅的翻译会因为重叠较多而具有不小的bleu值。



