class Features():


    def get_features(cls, image, relative=True):
        width = len(image[0])

        features = []
        for i in range(width):
            window = cls.get_window(image, i)
            cur_features = cls.features_for_window(window, relative)
            features.append(cur_features)

        return features


    def get_window(cls, image, pos):
        window = []

        for i in range(len(image)):
            if image[i][pos] == 0:
                window.append(1)
            else:
                window.append(0)

        return window



    def features_for_window(cls, window, relative=True):
        features = []

        features.append(cls.contour_top(window, relative))
        features.append(cls.contour_bottom(window, relative))
        features.append(cls.contour_height(features[0], features[1]))
        features.append(cls.black_amount(window, relative))
        features.append(cls.transitions(window, relative))

        return features



    def contour_top(cls, window, relative=True):
        pos = 0

        # calculate absolute position
        for i in range(len(window)):
            if window[i] == 1:
                pos = i
                break

        pos = len(window) - pos

        if relative:
            return pos / len(window)
        return pos



    def contour_bottom(cls, window, relative=True):
        pos = len(window)

        # calculate absolute position
        for i in range(len(window)):
            if window[i] == 1:
                pos = i

        pos = len(window) - pos

        if relative:
            return pos / len(window)
        return pos



    def contour_height(cls, contour_top, contour_bottom):
        return contour_top - contour_bottom



    def black_amount(cls, window, relative=True):
        black = 0

        for i in range(len(window)):
            if window[i] == 1:
                black += 1

        if relative:
            return black / len(window)
        return black



    def transitions(cls, window, relative=True):
        current = None
        transitions = 0

        for i in range(len(window)):
            if current is None:
                current = window[i]
            elif current != window[i]:
                current = window[i]
                transitions += 1

        if relative:
            return transitions / len(window)
        return transitions
