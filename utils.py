def readData(inputf):
    with open(inputf, encoding='utf-8') as inp:
        lines = inp.readlines()
    first = True
    data = []
    for line in lines:
        line = line.strip()
        if(line == '<|endoftext|>'):
            first = True
            continue
        elif first:
            data.append([])
            first = False
        data[-1].extend(line.split())
        data[-1].append('\n')
    return data

def substitute_with_UNK (data, n=1):
	tokens = {}
	for s in data:
		for w in s:
			if w not in tokens:
				tokens[w] = 1
			else:
				tokens[w] = tokens[w] + 1

	for s in range(len(data)):
		for w in range(len(data[s])):
			if tokens[data[s][w]] <= n:
				data[s][w] = 'UNK'
	return data
