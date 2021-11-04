# tyCooNN

# About
  tRNA expression Analesis using 1D-CNN


# Commands

  Usage: tyCooNN.py [OPTIONS] COMMAND [ARGS]...

   Options:
     --help  Show this message and exit.
   
   Commands:
     analysis
     evaluatetest
     makeparquetall
     makeparqueteach
     train

1.　Prepare training dataset for each tRNA
   - 1 . Read fast5, format(trimmed) and convert it to parquet file
     
    python tyCooNN.py makeparqueteach     
     
        -l, --tRNAlabel
        -i, --indir
        -o, --outpq
        -c, --takeCount,default=12000
        -p, --paramPath,default='settings.yaml'

     or

    python tyCooNN.py makeparquetall

        -ls, --listOfIOPath
        -p, --paramPath,default='settings.yaml'
              
2. train  CNN
    
    python tyCooNN.py train
        
        -i, --input
        -o, --outdir
        -e, --epoch,default=50
        -a, --data_argument,default=50        

3. Test accuracy using isolated tRNA data sets
     
    python tyCooNN.py evaluatetest
    
       -i, --input
       -o, --outdir
       -c, --csvout


4. Expression Analyses(Inference)
   
   - Inference using data of total tRNA
    
    python tyCooNN.py analysis   
   
       -p, --paramPath,default='settings.yaml'
       -i, --indir
       -c, --configdir
       -o, --outpath
       -f, --fasta



    
 


