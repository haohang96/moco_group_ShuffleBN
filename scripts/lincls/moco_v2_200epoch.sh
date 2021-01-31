python main_lincls.py \
--nomoxing \
--train_url=../moco_v2_200epoch \
--data_dir=/home/xuhaohang/toy_imagenet \
--usupv_lr=0.03 \
--usupv_batch=2 \
--pretrained_epoch=0 \
--init_lr=30. \
--batch_size=256 \
--wd=0. \
--selected_feat_id=17
#--resume=true \
