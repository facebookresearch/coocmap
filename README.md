# Accessing higher dimensions for unsupervised translation

Code for the paper [Accessing higher dimensions for unsupervised translation](https://arxiv.org/abs/2305.14200),
a simple and direct way to achieve unsupervised word translation that could conceivably be done in the 90s.

## Requirements

A short dependency list with comments in `requirements.txt`: tokenizers, numpy, cython,
pandas for data processing and wandb for experiment management.
fasttext is required for comparison experiments but not for the core method / demo usage.
The exact requirements are in `environment.yml`

## Demo usage

* Run `python setup.py build_ext --inplace` in `fast/` to build `fast/cooc_count.pyx`

* Download and process the raw data using Europarl English to Hungarian as the example
    * Download the compressed text data `wget https://www.statmt.org/europarl/v7/hu-en.tgz`
    * Extract plain text data: `tar xzf hu-en.tgz` to get `europarl-v7.hu-en.{hu, en}` 
    * shuffle the parallel data, for example
    ```
    shuf -o europarl-v7.hu-en.en < europarl-v7.hu-en.en
    shuf -o europarl-v7.hu-en.hu < europarl-v7.hu-en.hu
    ```

* Run with default parameters with `python test_coocmap.py`
    * The default demo setting uses 20MB of data and achieve an accuracy of `> 50%` in under 5min on 8 core CPUs x 2.4GHz
    * If not already, the evaluation data will be downloaded by `data.py` to `~/.cache/cooc/muse_dicts` (requires `wget`)
    for language pairs covered by [MUSE evaluation data](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries)
    * The default data paths and evaluation dictionary corresponds to this Europarl example, modify the default parameters accordingly to use other data source or language pairs

* In my testing, this demo got `54%` accuracy. The induced dictionaries are stored in `wandb/latest-run/files/dump/*.csv` among other log and run information.


## Code files

`match.py` contains the main methods vecmap and coocmap as well as methods for clip / drop and initialization. The main self-learn loop is `coocmapt`

`test_coocmap.py` runs the demo code, parameters are inline

`embeddings.py` is adopted from vecmap, with additions for comparison with other association methods

`data.py` contains most scripts to download and process data. Most data is saved in `~/.cache/cooc/` by default

`fast/` Cython code to count co-occurences faster than python.

`experiments/` contains wandb sweeps configs for experiments and their corresponding scripts

## License

The majority of coocmap is licensed under CC-BY-NC, however portions of the project are available under separate license terms: https://github.com/artetxem/vecmap/blob/master/embeddings.py is licensed under the GNU General Public License