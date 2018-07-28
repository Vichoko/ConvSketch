import numpy

from configuration_sketch import CLASSES_COUNT, TEST_EXAMPLES_PER_CLASS


def main():
    # load vectors; dims should be (50*100, 1024) and be ordered
    vectors_path = ''
    # class_bitmaps = numpy.load(vectors_path, 'r')
    feature_vectors = numpy.random.rand(TEST_EXAMPLES_PER_CLASS * CLASSES_COUNT, 1024)  # STUB todo: remove
    labels = []
    for label_idx in range(CLASSES_COUNT):
        # as test data is ordered by class, labels can be generaed this way
        labels += [label_idx] * TEST_EXAMPLES_PER_CLASS

    mean_avg_precision = calculate_mAP(feature_vectors, labels)
    print("mAP score: {}".format(mean_avg_precision))
    return mean_avg_precision


def calculate_mAP(feature_vectors, labels):
    """
    Calculate Mean Average Precition for given feature_vector, label pairs.
    :param feature_vectors: Array of feature vectors.
    :param labels: Array of correspondng label of each feature vector.
    :return:
    """
    # couple vector and labels
    vectors_n_labels = []
    for idx, example in enumerate(feature_vectors):
        vectors_n_labels.append((example, labels[idx]))

    avg_precisions = []
    for vector, label in vectors_n_labels:
        # for each vector, evaluate mAP
        ranking = sorted(vectors_n_labels,
                         key=lambda element: numpy.linalg.norm(
                             element[0] - vector))  # vectors are sorted from closer to farther to the ranking vector
        ranking = ranking[1:]  # discard the first one, as it's the same ranking vector

        # calculate mAP
        assertions = 0
        recall_acc = 0
        for idx, vector_label in enumerate(ranking):
            # iterate over the ranking calculating recall
            other_vector, other_label = vector_label
            if label == other_label:
                assertions += 1
                recall_acc += assertions / (idx + 1)
        # calc average precision of this ranking
        avg_precisions.append(recall_acc / assertions)

    # calc mean average precision over all rankings
    mean_avg_precision = numpy.mean(avg_precisions)
    return mean_avg_precision


if __name__ == "__main__":
    main()
