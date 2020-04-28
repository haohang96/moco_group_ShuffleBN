CUDA_VISIBLE_DEVICES=0,1 \
python main_lincls.py \
--nomoxing \
--train_url=../moco_v2 \
--data_dir=/root/toy_imagenet \
--usupv_lr=0.03 \
--usupv_batch=8 \
--pretrained_epoch=0 \
--init_lr=30. \
--batch_size=8 \
--wd=0. \
#--resume=true \
