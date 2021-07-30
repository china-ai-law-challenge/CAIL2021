def label2idx(result):
    results = result.copy()
    for i, result in enumerate(results):
        results[i] = int(result.split('/')[-1][1:])

    return results


def get_fact(content):
    breakers = []
    fact_starts = []

    for i, c in enumerate(content):
        if c[0] == '【' and c[-1] == '】':
            breakers.append(i)
            if '查明' in c:
                fact_starts.append(i)

    fact = []
    for start in fact_starts:
        end = breakers[breakers.index(start)+1]
        fact += content[start:end]

    return ''.join(fact)


def get_level3labels(tree):
    level3labels = []
    for t in tree:
        level1 = t['value']
        children1 = t['children']
        for child1 in children1:
            level2 = child1['value']
            children2 = child1['children']
            for child2 in children2:
                level3 = child2['value']
                level3labels.append('/'.join([level1, level2, level3]))
    return level3labels
