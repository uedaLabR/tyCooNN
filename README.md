# tyCooNN

# tRNA expression Analesis using 1D-CNN

1. Train with isolated tRNA data
   - preprocess 1 . Read fast5, format(trimmed) and convert it to parquet file
   - preprocess 2.  training  train CNN
   

2. Expression Analyses(Inference)
   
   - Inference using data of total tRNA

# Command

  ## Step 1: Make format parquet file
        python preprocess.py parquet --inp inpfile --out parquetOutFile
        python preprocess.py parquet --inp inpdir  --out parquetOutFile
  	e.g. python preprocess.py parquet --inp $inpfile --out data/pq/ala2/ala2.pq

  ## Step 2: Trim by HMM
  	python preprocess.py trim --inp data/pq/ala2/ala2.pq --out data/trim/ala2/ala2.pq --index 0

  ## Step 3.1: For Training
	### Using 2 GPU
  	export CUDA_VISIBLE_DEVICES="0,1"
  	python tycon.py train --inp data/trimAll  --out data/training  --extn .parquet --ngpu 2 \
        	--epoch 500 --limit 25000 --batch 64 --ncpu 4 --memgpu 20 --lr 0.0008
	### Using 1 GPU
  	export CUDA_VISIBLE_DEVICES="0"
  	python tycon.py train --inp data/trimAll  --out data/training  --extn .parquet --ngpu 1 \
        	--epoch 500 --limit 25000 --batch 64 --ncpu 4 --memgpu 20 --lr 0.0008
	### Using CPUs
  	export CUDA_VISIBLE_DEVICES=""
  	python tycon.py train --inp data/trimAll  --out data/training  --extn .parquet --ngpu 0 \
        	--epoch 500 --limit 25000 --batch 64 --ncpu 4 --memgpu 20 --lr 0.0008

  ## Step 3.2: For Testing
  	python tycon.py test --inp data/trimMix.parquet --out data/infer.csv --weight data/training/model_trna100.h5 \
        	--ngpu 1 --ncpu 4 --memgpu 20 --lr 0.0008 --ncl 47 --trna trna.list



