#   MMSAR - classical causal filters M1 to M4
#        Oumar Ibrahim U22200741

# every filter only looks at the current frame and the frames before it
# at the start of a clip the window is just the frames seen so far


# M1 raw thresholding
class BaselineFilter:
    def decide(self, confidence, tau):
        out = []
        for c in confidence:
            if c >= tau:
                out.append(1)
            else:
                out.append(0)
        return out


# M2 avrage of the last w frames
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size

    def decide(self, confidence, tau):
        out = []
        for i in range(len(confidence)):
            start = i - self.window_size + 1
            if start < 0:
                start = 0
            window = confidence[start : i + 1]

            total = 0
            for c in window:
                total += c
            mean = total / len(window)

            if mean >= tau:
                out.append(1)
            else:
                out.append(0)
        return out


# M3 median of the last w frames, kills single frame spikes
class MedianFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size

    def decide(self, confidence, tau):
        out = []
        for i in range(len(confidence)):
            start = i - self.window_size + 1
            if start < 0:
                start = 0
            window = sorted(confidence[start : i + 1])

            n = len(window)
            if n % 2 == 1:
                med = window[n // 2]
            else:
                med = (window[n // 2 - 1] + window[n // 2]) / 2

            if med >= tau:
                out.append(1)
            else:
                out.append(0)
        return out


# M4 alert when at least `hits` of the last w frames pass the threshlod
class HistoryConsensusFilter:
    def __init__(self, window_size=5, hits=3):
        self.window_size = window_size
        self.hits = hits

    def decide(self, confidence, tau):
        out = []
        for i in range(len(confidence)):
            start = i - self.window_size + 1
            if start < 0:
                start = 0

            count = 0
            for c in confidence[start : i + 1]:
                if c >= tau:
                    count += 1

            if count >= self.hits:
                out.append(1)
            else:
                out.append(0)
        return out
