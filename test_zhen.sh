export PYTHONPATH=./rl4nmt/:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=7
binFile=./rl4nmt/tensor2tensor/bin

beamsize=${2:-6}
PROBLEM=translate_zhen_wmt17
MODEL=transformer
HPARAMS=zhen_wmt17_transformer_rl_delta_setting_random

DATA_DIR=../transformer_data/zhen
USR_DIR=../rl4nmt/zhen_wmt17
ROOT_MODEL=./rl4nmt/model/${HPARAMS}

for ii in {100000..120000..500}; do
  tmpdir=${ROOT_MODEL}_${ii}
  rm -rf $tmpdir
  mkdir -p $tmpdir
  cp ${ROOT_MODEL}/model.ckpt-${ii}* $tmpdir/
  cd $tmpdir
  touch checkpoint
  echo model_checkpoint_path: \"model.ckpt-${ii}\" >> checkpoint
  echo all_model_checkpoint_paths: \"model.ckpt-${ii}\" >> checkpoint
  cd ../transformer_data/zhen  # test data path
  cp $DATA_DIR/test.zh $tmpdir/
  echo ${ii}

  ${binFile}/t2t-decoder \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS \
    --output_dir=$tmpdir \
    --decode_hparams="beam_size=${beamsize},alpha=1.1,batch_size=32" \
    --decode_from_file=$tmpdir/test.zh \
    --worker_gpu=1
done
