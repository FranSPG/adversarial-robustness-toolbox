
# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""

| Paper link:
"""
from PIL import Image
from typing import List
import numpy as np
import hashlib


class Blacklight:
    """

    """
    def __init__(self, window_size: int = 20,
                 sliding_step: int = 1,
                 n_hashes_match_threshold: int = 20):
        if window_size < n_hashes_match_threshold:
            raise ValueError("Number of hashes to match must be less than window size")

        self.window_size = window_size
        self.sliding_step = sliding_step
        self.n_hashes_match_threshold = n_hashes_match_threshold
        self.hashes = []

    def scan(self, x_adv):
        x_adv = self.process_input(x_adv)
        x_adv_vectors = self.get_x_adv_vectores(x_adv)
        decision, n_intersection = self.compare_overlap(x_adv_vectors)
        return decision, n_intersection

    def compare_overlap(self, x_adv_vectors):
        for i in range(len(self.hashes)):
            try:
                hashes = self.hashes[i][:self.n_hashes_match_threshold]
                n_intersection = len(list(set(hashes) & set(x_adv_vectors)))
            except Exception as e:
                print(e)
                pass

        return n_intersection >= self.n_hashes_match_threshold, n_intersection



    def get_x_adv_vectores(self, x_adv):
        N_windows = int((x_adv.shape[0] - self.window_size + self.sliding_step) / self.sliding_step)
        vectors = []
        for i in range(0, N_windows):
            x_adv_slice = x_adv[i:self.window_size]
            x_adv_slice_hash = hashlib.sha3_256(x_adv_slice).hexdigest()
            vectors.append(x_adv_slice_hash)
        return sorted(vectors[:self.n_hashes_compare])

    def process_input(self, x_adv: str) -> np.ndarray:
        if isinstance(x_adv, str):
            x_adv = Image.open(x_adv)
        elif isinstance(x_adv, np.ndarray):
            x_adv = Image.fromarray(x_adv)
        x_adv = x_adv.quantize(256)
        return x_adv.__array__().flatten('C')
