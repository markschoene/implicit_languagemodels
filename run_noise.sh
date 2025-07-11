for ema in 0.2 0.1 
do
    for noise_mode in additive multiplicative
    do
        for state_noise_db in 0.0 20.0 30.0 40.0
        do
            for latent_noise_db in 0.0 10.0 20.0 30.0
            do
                echo "Running noise.py with noise_mode=$noise_mode, state_noise_db=$state_noise_db, latent_noise_db=$latent_noise_db, ema_alpha=$ema"
                # python noise.py --model_name hf_models/mamba2-130m-explicit --batches 2 --batch_size 128 \
                #     --noise_mode $noise_mode --state_noise_db $state_noise_db --latent_noise_db $latent_noise_db
                python noise.py --model_name hf_models/mamba2-130m-implicit --batches 2 --batch_size 128 \
                    --noise_mode $noise_mode --state_noise_db $state_noise_db --latent_noise_db $latent_noise_db \
                    --ema_alpha $ema
            done
        done
    done
done