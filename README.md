# Reinforcement Learning for Neural Machine Translation (RL4NMT)
EMNLP 2018: A Study of Reinforcement Learning for Neural Machine Translation
```
@inproceedings{wu2018study,
  title={A Study of Reinforcement Learning for Neural Machine Translation},
  author={Wu, Lijun and Tian, Fei and Qin, Tao and Lai, Jianhuang and Liu, Tie-Yan},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={3612--3621},
  year={2018}
}
```

# RL4NMT based on Transformer
Please first get familar with the basic Tensor2Tensor project: https://github.com/tensorflow/tensor2tensor.

The tesorflow version is 1.4, and the tensor2tensor version is 1.2.9. 

Take WMT17 Chinese-English translation as example: 

Different training strategies are provided.

* Different RL training strategies for NMT, evaluated on bilingual dataset. <br>
(1) HPARAMS=zhen_wmt17_transformer_rl_total_setting: terminal reward + beam search <br>
(2) HPARAMS=zhen_wmt17_transformer_rl_delta_setting: reward shapping + beam search <br>
(3) HPARAMS=zhen_wmt17_transformer_rl_delta_setting_random: reward shapping + multinomial sampling <br>
(4) HPARAMS=zhen_wmt17_transformer_rl_total_setting_random: terminal reward + multinomial sampling <br>
(5) HPARAMS=zhen_wmt17_transformer_rl_delta_setting_random_baseline: reward shaping + multinomial sampling + reward baseline <br>
(6) HPARAMS=zhen_wmt17_transformer_rl_delta_setting_random_mle: reward shapping + multinomial sampling + objectives combination

* Different monolingual data combination traininig in RL4NMT <br>
(1) zhen_src_mono: source monolingual data RL training based on bilingual data MLE model <br>
(2) zhen_tgt_mono: target monolingual data RL training based on bilingual data MLE model <br>
(3) zhen_src_tgt_mono: sequential mode [target monolingual data RL trianing based on (bilingual + source monolingual data) MLE model] <br>
(4) zhen_tgt_src_mono: sequential mode [source monolingual data RL training based on (bilinugal + target monolingual data) MLE model] <br>
(5) zhen_bi_src_tgt_mono: unified model

Supports MRT (minimum risk training) for NMT. 
