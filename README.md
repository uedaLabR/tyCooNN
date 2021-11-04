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
![image](https://user-images.githubusercontent.com/70622849/140273121-1d7312ee-d1e9-4891-aa3d-dc2a4a86853d.png)
  - Read fast5, format(trimmed) and convert it to parquet file
     
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
  - train CNN model using isolated tRNA data sets and save weight
     
    python tyCooNN.py train
        
        -i, --input
        -o, --outdir
        -e, --epoch,default=50
        -a, --data_argument,default=50        

3. Test accuracy using isolated tRNA data sets
  - test accuracy of model using isolated tRNA data sets
     
    python tyCooNN.py evaluatetest
    
       -i, --input
       -o, --outdir
       -c, --csvout


4. Expression Analyses(Inference)
   - Inference using data of total tRNA and classify
     
    
    python tyCooNN.py analysis   
   
       -p, --paramPath,default='settings.yaml'
       -i, --indir
       -c, --configdir
       -o, --outpath
       -f, --fasta



    
 


