* snakemake --dry-run export_genotypes → confirm rule graph is valid
* Run prepare_data.py on a small subset (1 chromosome, 50 individuals) to verify HDF5 shape and label alignment
* Confirm train/val/test split has zero individual overlap: assert len(set(train_ids) & set(test_ids)) == 0
* Overfit on 100 windows from 1 chromosome to verify training loop and loss decreases to ~0
* On Colab: verify CUDA is available (torch.cuda.is_available()) and batch moves to GPU before launching full training
* LAI sanity check: on a simulated 2-way admixed individual, the predicted class should match the source population on each side of the crossover point