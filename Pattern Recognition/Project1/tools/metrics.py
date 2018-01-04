def accuracy(expected, actual):
    errors = 0
    for i, expected_label in enumerate(expected):
        actual_label = actual[i]

        if actual_label != expected_label:
            errors += 1

    return round(((len(expected) - errors) / len(expected)) * 100, 2)
