export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"

dataset=videoartgs
subset=sapien
scenes=('100481' '101284' '101287' '101808' '101908' '103015' '103811' '10489' '10655' '168' '25493' '30666' '31249' '45194' '45503' '45612' '47648' '8961' '9016' '1280')

seed=0
model_name=final

iter=20000
for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python eval_track.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --iteration ${iter} \

done
