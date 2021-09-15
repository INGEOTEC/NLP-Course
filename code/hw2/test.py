import json

if __name__ == '__main__':
    from performance import performance
    from hw2 import ColgateSBD
    perf = performance(ColgateSBD())

    num_sent = sum([x[-1] for x in perf])
    tests = [dict(name="Number of sentences",
                  max_score=1,
                  score=int(num_sent))]
    for fname, *per in perf:
        for name, value in zip(["Accuracy", "Recall", "Precision", "F1"],
                                per[0]):
            _ = dict(name=fname + ": " + name,
                     max_score=1,
                     score=float(value))
            tests.append(_)
    output = dict(visibility="visible",
                  tests=tests)
    _ = '/autograder/results/results.json'
    # _ = 'results.json'
    with open(_, 'w') as f:
        json.dump(output, f)