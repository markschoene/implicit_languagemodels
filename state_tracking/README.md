# State Tracking
The experiments were inspired by [Merrill et al., "The Illusion of State"](https://arxiv.org/abs/2404.08819) (check out [their code](https://github.com/jopetty/word-problem)).

## Setup
Our implementation is based on the [abstract_algebra package](https://github.com/alreich/abstract_algebra).
```bash
pip install git+https://github.com/alreich/abstract_algebra.git@bbe5c66b1a1b34b08d2c8f75647ae0062c954dd5
```
## Generate the dataset
A single tensor dataset of shape `(num_samples, batch_size, length)` can be generated via
```bash
python dataset.py --dirname data --prob 0.5 --length 256 --mode train
```
For our paper, we generated multiple splits to test how many hard tokens need to be seed during training.
```bash
# generate training tokens
for prob in 0.0 0.5 0.75 0.9 0.95 0.98 1.0
    do python dataset.py --dirname data --prob $prob --length 256 --mode train
done

# generate test tokens
for prob in 0.5 0.9
    for len in 256 1024
        do python dataset.py --dirname data --prob $prob --length $len --mode test
    done
done
```
Or to speed up the process
```bash
# generate training tokens
parallel python dataset.py --dirname data --prob {} --length 256 --mode train ::: 0.0 0.5 0.75 0.9 0.95 0.98 1.0

# generate test tokens
parallel python dataset.py --dirname data --prob {1} --length {2} --mode test ::: 0.5 0.9 ::: 256 1024
```

## Variants of the dataset
Modify `dataset.py` for your needs. E.g. to run a clean $A_5$ state-tracking task similar to Merill et al., simply change the monoid in line 222 to `
```python
monoid = identity_monoid()
```

Larger algebras can be constructed e.g. by direct products
```python
m = aperiodic_semigroup()
monoid = m * m * m
```

Make sure to change the datatype `--dtype` accordingly when the vocabulary size grows larger than 256. 
The data generation script will throw a warning in this case.
Also make sure to change the vocab size in the model configs.