class Evaluation:
    def __init__(self, all_elements, query_element, retrieved_elements):
        self.all_elements = all_elements

        self.query_label = query_element.label
        self.retrieved_elements = retrieved_elements
        self.retrieved_labels = list(map(lambda i: i.label, retrieved_elements))

    def precision(self):
        true_positives_count = len(self.true_positives())
        false_positives_count = len(self.false_positives())

        if true_positives_count == 0 and false_positives_count == 0:
            return 0

        return true_positives_count / (true_positives_count + false_positives_count)

    def recall(self):
        true_positives_count = len(self.true_positives())
        false_negatives_count = len(self.false_negatives())

        if true_positives_count == 0 and false_negatives_count == 0:
            return 0

        return true_positives_count / (true_positives_count + false_negatives_count)

    def true_positives(self):
        return list(filter(lambda x: x == self.query_label, self.retrieved_labels))

    def false_positives(self):
        return list(filter(lambda x: x != self.query_label, self.retrieved_labels))

    def false_negatives(self):
        all_elements_with_query_label = list(filter(lambda x: x.label == self.query_label, self.all_elements))
        return list(filter(lambda x: x not in self.retrieved_elements, all_elements_with_query_label))
