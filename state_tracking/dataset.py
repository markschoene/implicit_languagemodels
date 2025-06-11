import math
import os

import torch
from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
    make_finite_algebra,
)
from tqdm import tqdm


def join_paths(base, path):
    if isinstance(path, str):
        return os.path.join(base, path)
    else:
        return [os.path.join(base, f) for f in path]


def generate_group(g: str) -> FiniteAlgebra:
    """Generate an group from a string identifier."""
    id = g[0]
    val = int(g[1:])
    if id == "S":
        return generate_symmetric_group(val)
    elif id == "Z":
        return generate_cyclic_group(val)
    elif id == "A":
        s_n = generate_symmetric_group(val)
        a_n = s_n.commutator_subalgebra()
        a_n.name = g
        return a_n
    else:
        raise ValueError("Group must be one of S, Z, or A")


def aperiodic_semigroup():
    """
    An explicit example of a semigroup that defines the star-free regular language
    over a and b that does not contain the substring aa.

    Elements:
    E: strings that end in b or the empty string
    A: strings that end in a
    B: strings that cannot be extended to violate the language
    X: strings that contain aa
    """
    cayley_table = [
        # E   A   B   X
        ["E", "A", "B", "X"],  # E
        ["A", "X", "X", "X"],  # A
        ["B", "A", "B", "X"],  # B
        ["X", "X", "X", "X"],  # X
    ]
    return make_finite_algebra(
        "M",
        "semigroup that defines the star-free regular language over a and b that does not contain the substring aa",
        ["E", "A", "B", "X"],
        cayley_table,
    )


def identity_monoid():
    """
    An identity monoid with a single element that acts as the identity.
    This is useful for testing purposes where we want to ensure that the
    group operation does not change the elements.
    """
    return make_finite_algebra(
        "I",
        "identity monoid",
        ["e"],
        [["e"]],
    )


class GroupDataset(torch.utils.data.Dataset):
    def __init__(self, group, monoid, sample_length, num_samples, p=1):
        """
        A group dataset that constructs the direct product G x M of a group G and a monoid M.

        Attributes:
            group (algebra): The group G for G x M.
            monoid (algebra): The monoid M for G x M.
            sample_length (int): The length of each sample generated.
            num_samples (int): The total number of samples to generate.
            p (float): probability of sampling the identity. I.e. p = 1 is the monoid operation only.

        Methods:
            __len__(): Returns the total number of samples in the dataset.
            __getitem__(index): Returns a specific sample from the dataset given an index.
        """
        self.group = group
        self.monoid = monoid
        self.product = self.direct_product()
        self.sample_length = sample_length
        self.num_samples = num_samples
        self.vocab_size = len(self.product.elements)
        self.p = p

    def direct_product(self):
        cache_dir = os.path.dirname(os.path.abspath(__file__))
        cache_file = os.path.join(cache_dir, f"{self.group.name}_cache.json")
        if os.path.isfile(cache_file):
            print(f"Reading product algebra from {cache_file}")
            return make_finite_algebra(cache_file)
        else:
            print("Constructing product algebra")
            prod_algebra = self.group * self.monoid
            with open(cache_file, "w") as f:
                f.write(prod_algebra.dumps())
            print(f"Wrote algebra JSON string to {cache_file}")
            return prod_algebra

    def product_index(self, idx_a, idx_b):
        a = self.group.elements[idx_a]
        b = self.monoid.elements[idx_b]
        element = self.join_elements(a, b)
        return self.product.elements.index(element)

    def product_op(self, idx_a, idx_b):
        a = self.product.elements[idx_a]
        b = self.product.elements[idx_b]
        c = self.product.op(a, b)
        return self.product.elements.index(c)

    def join_elements(self, a, b):
        return ":".join([a, b])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a random sequence of group elements and their prefix products as targets"""
        mask = torch.rand(self.sample_length) > self.p
        group_sample = torch.randint(0, len(self.group.elements), (self.sample_length,), dtype=torch.long)
        group_sample = mask * group_sample
        monoid_sample = torch.randint(0, len(self.monoid.elements), (self.sample_length,), dtype=torch.long)

        samples = torch.zeros(self.sample_length, dtype=torch.long)
        targets = torch.zeros(self.sample_length, dtype=torch.long)
        first_element = self.product_index(group_sample[0], monoid_sample[0])
        samples[0] = first_element
        targets[0] = first_element
        for j in range(1, self.sample_length):
            idx_a = targets[j - 1]
            idx_b = self.product_index(group_sample[j], monoid_sample[j])
            samples[j] = idx_b
            targets[j] = self.product_op(idx_a, idx_b)

        return samples, targets

    def dump(self, path, dtype=torch.long):
        min_bits = math.log2(self.vocab_size)
        print(f"Vocab size is {self.vocab_size}, which requires at least {min_bits} bits")

        inputs = torch.zeros(self.num_samples, self.sample_length, dtype=torch.long)
        targets = torch.zeros(self.num_samples, self.sample_length, dtype=torch.long)
        for i in tqdm(range(self.num_samples)):
            x, y = self.__getitem__(i)
            inputs[i] = x
            targets[i] = y
        torch.save(
            {
                "inputs": inputs.to(dtype=dtype),
                "targets": targets.to(dtype=dtype),
                "vocab_size": torch.LongTensor([self.vocab_size]),
            },
            path,
        )


class HuggingFaceDataset(torch.utils.data.Dataset):
    """
    A HF compatible dataset to train with the Trainer class.
    """

    def __init__(self, path):
        self.data = torch.load(path, weight_only=True)
        self.inputs = self.data["inputs"]
        self.targets = self.data["targets"]
        self.vocab_size = self.data["vocab_size"].item()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.targets[idx]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--dirname", type=str, help="base directory for saving the datasets")
    parser.add_argument("-p", "--prob", type=float, help="probability of sampling from the group")
    parser.add_argument("-l", "--length", type=int, help="Sequence length of the samples")
    parser.add_argument("-m", "--mode", type=str, choices=["train", "test"], help="Training or test split")
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "uint16", "uint32", "int32"],
        help="Data type of the samples",
    )
    args = parser.parse_args()

    # create the base directory if it does not exist
    if not os.path.exists(args.dirname):
        os.makedirs(args.dirname)

    # Define the group to use. For hard state tracking, choose a non-solvable group such as 'A5' or 'S5'.
    group_name = "A5"  # Example group name, can be changed
    group = generate_group(group_name)

    # Define the monoid to use. Here we use a specific semigroup that defines a star-free regular language.
    # To construct more complex monoids, you can define your own Cayley table or expand via direct products.
    # Example (only A5): monoid = identity_monoid()
    # Example (complex): monoid = aperiodic_semigroup() * aperiodic_semigroup() * aperiodic_semigroup()
    monoid = aperiodic_semigroup()

    num_samples = {
        "train": 512 * 16384,
        "test": 4096,
    }
    mode = args.mode
    length = args.length
    p = args.prob
    print(f"Generating mode {mode} - L={length} with prob {p}")

    dtype = getattr(torch, args.dtype)
    dataset = GroupDataset(group, monoid, length, num_samples[mode], p)
    save_path = os.path.join(args.dirname, f"{mode}_{group_name}_L{length}_P{int(100 * p):03d}.bin")
    dataset.dump(save_path)
