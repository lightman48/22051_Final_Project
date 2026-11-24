
import numpy as np

# NODE TYPE
# {Time, Level}
class AutoParameter:
    def __init__(self, nodes: list[list], samplerate):
        self.samplerate = samplerate
        self.nodes = nodes

    def process(self) -> np.ndarray:
        period = self.nodes[-1][0]
        output = np.zeros(int(self.samplerate * period))

        is_first_node = True
        previous_node = self.nodes[0]

        for node in self.nodes:
            if is_first_node:
                is_first_node = False
                previous_node = self.nodes[0]
                continue

            t0 = previous_node[0]
            t1 = node[0]

            y0 = previous_node[1]
            y1 = node[1]

            dt = 1 / self.samplerate
            nsamples = int((t1 - t0) * self.samplerate)
            slope = (y1 - y0) / (t1 - t0)

            sample_0 = int(t0 * self.samplerate)
            for i in range(nsamples):
                output[sample_0 + i] = y0 + slope * (i * dt)

            previous_node = node
        return output




            
            
