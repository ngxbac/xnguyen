for subj in 2 3 4 5 6 7 8 ; do
    CUDA_VISIBLE_DEVICES=${subj} python llma.py --subj ${subj}
done