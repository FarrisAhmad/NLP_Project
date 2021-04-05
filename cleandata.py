import nltk

with open('poetrycleaned.txt', encoding='utf-8') as inp:
    lines = inp.readlines()
output = open('poetryextracleaned.txt','w',encoding='utf-8')
sentences = []
for line in lines:
    if '<|endoftext|>' in line:
        output.write(line)
    else:
        words = nltk.word_tokenize(line)
        newwords = [word for word in words if word.isalnum()]
        for word in newwords:
            output.write(word+" ")
        output.write("\n")