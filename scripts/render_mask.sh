export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"

dataset=artgs
subset=sapien
scenes=('101908' '101917' '10211' '102255' '103111' '103706_eevee' '103706_rotate' '103776_eevee' '10537' '10537_rotate' '10905' '10905_bg' '25493' '31249' '45503' '47648')
# scenes=('101908')

seed=0
model_name=base_tl0.5
res=1
iter=10000

for scene in ${scenes[@]};do
    # model_path=outputs/best/${dataset}/${scene}
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python render_mask.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --resolution ${res} \
        --iteration ${iter} \
        --visualize \

done
