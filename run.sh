#'''
export CUDA_VISIBLE_DEVICES=0
#'''
model=Sarnet

item=deer
txt=${item}_new_data
channel=64
sub=16
seed=123
mode=eccv
name2=radon_seed_${seed}

out_folder=output_result
#'''
python runner.py --arch ${model}\
       	--out ${model}_${mode}_${item}_${channel}_${sub}_${name2}\
       	--in ${txt}\
       	--phase train\
	--channel_num ${channel}\
	--self_reduction 1\
	--channel_reduction 1\
	--channel_att eca\
	--k 3\
	--lr 1e-3\
	--seed ${seed}\
	--mode ${mode}\
	--sub ${sub}
        #--weights $1 #Unet_Freq_eca_att_32_8_3_no_datt/2021-07-10T03_33_38.873310_Unet_Freq_CA_att/checkpoints/ckpt-epoch-3000.pth.tar
#'''
python runner.py --arch ${model}\
	--out ${out_folder}/${model}_${mode}_${item}_${channel}_${sub}_${name2}\
	--in ${txt}\
	--phase best\
	--channel_num ${channel}\
	--self_reduction 1\
	--channel_reduction 1\
	--channel_att eca\
	--k 3\
	--sub ${sub}\
	--mode ${mode}\
	--weights ${model}_${mode}_${item}_${channel}_${sub}_${name2}/checkpoints/best-ckpt.pth.tar


python runner.py --arch ${model}\
	--out ${out_folder}/${model}_${mode}_${item}_${channel}_${sub}_${name2}\
	--in ${txt}\
	--phase 2000\
	--channel_num ${channel}\
	--self_reduction 1\
	--channel_reduction 1\
	--channel_att eca\
	--k 3\
	--sub ${sub}\
	--mode ${mode}\
	--weights ${model}_${mode}_${item}_${channel}_${sub}_${name2}/checkpoints/ckpt-epoch-3000.pth.tar

python runner.py --arch ${model}\
	--out ${out_folder}/${model}_${mode}_${item}_${channel}_${sub}_${name2}\
	--in ${txt}\
        --phase 3000\
	--channel_num ${channel}\
	--self_reduction 1\
	--channel_reduction 1\
	--channel_att eca\
	--k 3\
	--sub ${sub}\
	--mode ${mode}\
	--weights ${model}_${mode}_${item}_${channel}_${sub}_${name2}/checkpoints/ckpt-epoch-3000.pth.tar
#'''
