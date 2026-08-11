export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"

dataset=videoartgs
subset=sapien
scenes=('100481' '101284' '101287' '101808' '101908' '103015' '103811' '10489' '10655' '168' '25493' '30666' '31249' '45194' '45503' '45612' '47648' '8961' '9016' '1280')
scenes=('100481')

# subset=realscan
# scenes=('cab1' 'chair_1r' 'mac_1r' 'microwave_ego' 'cab_1r_1p' 'coffeemachine_2r' 'microwave_1r')

# dataset=v2a
# scenes=('100068_joint_0_bg_view_0' '100071_joint_0_bg_view_1' '100072_joint_0_bg_view_0' '100087_joint_0_bg_view_0' '100092_joint_0_bg_view_0' '100106_joint_0_bg_view_1' '100128_joint_0_bg_view_1' '100133_joint_0_bg_view_0' '10040_joint_1_bg_view_0' '100664_joint_0_bg_view_1')
# scenes=('10143_joint_0_bg_view_1' '101886_joint_0_bg_view_0' '101948_joint_0_bg_view_1' '10306_joint_1_bg_view_1' '103273_joint_1_bg_view_1' '103283_joint_1_bg_view_1' '103351_joint_0_bg_view_0' '103490_joint_0_bg_view_0' '103528_joint_0_bg_view_1' '10356_joint_0_bg_view_1' '10373_joint_0_bg_view_0' '103778_joint_0_bg_view_0' '10495_joint_0_bg_view_1' '10558_joint_0_bg_view_0' '10559_joint_0_bg_view_1' '10567_joint_0_bg_view_0' '10867_joint_0_bg_view_1' '10895_joint_1_bg_view_0' '10973_joint_0_bg_view_1' '11304_joint_0_bg_view_0' '11304_joint_1_bg_view_0' '11700_joint_0_bg_view_1' '12071_joint_0_bg_view_0' '12531_joint_0_bg_view_1' '12552_joint_0_bg_view_0' '12552_joint_0_bg_view_1' '12562_joint_0_bg_view_1' '12565_joint_1_bg_view_0' '19179_joint_1_bg_view_1' '19855_joint_0_bg_view_1' '19898_joint_1_bg_view_1' '19898_joint_3_bg_view_0' '19898_joint_4_bg_view_1' '20745_joint_0_bg_view_1' '20985_joint_1_bg_view_0' '22241_joint_0_bg_view_1' '22339_joint_0_bg_view_1' '22367_joint_0_bg_view_0' '22367_joint_2_bg_view_0' '22433_joint_0_bg_view_0' '22433_joint_0_bg_view_1' '22433_joint_1_bg_view_0' '23372_joint_1_bg_view_1' '23724_joint_0_bg_view_1' '23724_joint_2_bg_view_0' '23807_joint_1_bg_view_1' '26525_joint_0_bg_view_1' '26608_joint_0_bg_view_1' '26657_joint_1_bg_view_0' '27267_joint_0_bg_view_1' '35059_joint_0_bg_view_0' '40453_joint_1_bg_view_1' '41083_joint_3_bg_view_1' '41510_joint_1_bg_view_1' '44781_joint_0_bg_view_1' '44817_joint_1_bg_view_1' '44962_joint_2_bg_view_1' '45001_joint_1_bg_view_0' '45132_joint_2_bg_view_0' '45146_joint_0_bg_view_0' '7265_joint_0_bg_view_0' '7265_joint_0_bg_view_1' '9987_joint_0_bg_view_1')

seed=0
model_name=final
res=1
iter=20000
for scene in ${scenes[@]};do
    # model_path=outputs/best/${dataset}/${scene}
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python render.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --resolution ${res} \
        --iteration ${iter} \
        --white_background \
        
done
