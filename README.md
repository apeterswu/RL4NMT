Reinforcement Learning for Neural Machine Translation

Dataset and models will be relased if needed. 

# Reinforcement Learning for Neural Machine Translation (RL4NMT) based on Transformer

If you use this code about cnn, please cite:
```
@inproceedings{gehring2017convs2s,
  author    = {Lijun, Wu and Fei, Tian and Tao, Qin and Jianhuang, Lai and Tie-Yan, Liu},
  title     = "{A Study of Reinforcement Learning for Neural Machine Translation}",
  booktitle = {Proc. of EMNLP},
  year      = 2018,
}
```

Take WMT17 Chinese-English translation as example: 

Different training strategies are provided.

1. Different RL training strategies for NMT, evaluated on bilingual dataset. <br>
(1) HPARAMS=zhen_wmt17_transformer_rl_total_setting: terminal reward + beam search <br>
(2) HPARAMS=zhen_wmt17_transformer_rl_delta_setting: reward shapping + beam search <br>
(3) HPARAMS=zhen_wmt17_transformer_rl_delta_setting_random: reward shapping + multinomial sampling <br>
(4) HPARAMS=zhen_wmt17_transformer_rl_total_setting_random: terminal reward + multinomial sampling <br>
(5) HPARAMS=zhen_wmt17_transformer_rl_delta_setting_random_baseline: reward shaping + multinomial sampling + reward baseline <br>
(6) HPARAMS=zhen_wmt17_transformer_rl_delta_setting_random_mle: reward shapping + multinomial sampling + objectives combination

2. Different monolingual data combination traininig in RL4NMT <br>
(1) zhen_src_mono: source monolingual data RL training based on bilingual data MLE model <br>
(2) zhen_tgt_mono: target monolingual data RL training based on bilingual data MLE model <br>
(3) zhen_src_tgt_mono: sequential mode [target monolingual data RL trianing based on (bilingual + source monolingual data) MLE model] <br>
(4) zhen_tgt_src_mono: sequential mode [source monolingual data RL training based on (bilinugal + target monolingual data) MLE model] <br>
(5) zhen_bi_src_tgt_mono: unified model

Supports MRT (minimum risk training) for NMT. 
