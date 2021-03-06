# FMRL

This repository contains code and scripts to recreate the task-similarity, quadratic growth, and deep learning experiments from the FMRL paper (citation below).

* task-similarity: <tt>similarity.py</tt>
* quadratic growth: <tt>growth.py</tt>
* deep learning: <tt>.sh</tt> scripts in <tt>reptile</tt> directory (edited clone of OpenAI's [Reptile codebase](https://github.com/openai/supervised-reptile))
* MiniWiki dataset: <tt>.tar.gz</tt> file in <tt>data/</tt>
* MiniWiki experiments: <tt>online.py</tt> (Dependencies: scikit-learn, cvxpy, scipy, numpy, torch, h5py, matplotlib, [text_embedding](https://github.com/NLPrinceton/text_embedding))

Citation:
  
    @inproceedings{khodak2019fmrl,
      title={Provable Guarantees for Gradient-Based Meta-Learning},
      author={Khodak, Mikhail and Balcan, Maria-Florina and Talwalkar, Ameet},
      booktitle={Proceedings of the 36th International Conference on Machine Learning,},
      year={2019}
    }
