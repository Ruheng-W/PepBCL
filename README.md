# PepBCL: a deep learning-based model to predict protein-peptide binding residues



## How to use it

To use it, firstly, you need to download pytorch_model.bin file from the following URL https://huggingface.co/Rostlab/prot_bert_bfd/blob/main/pytorch_model.bin. And put pytorch_model.bin file into prot_bert_bfd directory.


The main program in the train folder protBert_main.py file. You could change the load_config function to achieve custom training and testing, such as modifying datasets, setting hyperparameters and so on. File protBert_main.py has detail notes.


The project is mainly implemented through **Pytorch** and **sklearn**. See requirements.txt for details of dependent packages.


To facilitate the use of our method, we establish an online predictive platform as the implementation of the proposed PepBCL, which is now available at http://server.wei-group.net/PepBCL/.

## To load model parameters from user-saved state_dict file

Users can download our saved model parameters of PepBCL from the following link: https://drive.google.com/drive/folders/1UTZxnR34UaUryKkXRaM0ts3ufjVQtCas. The directory contains two parameter files corresponding to two datasets (Dataset1 and Dataset2). 

For using our saved model parameters, users firstly need to unzip the model parameter file. Then they could easily load the model parameter by the following codes:
```python
model = prot_bert.BERT(config)                       // model instantiation
model_dict = torch.load('THE FILE PATH')['model']    // load the state_dict from file path
model.load_state_dict(model_dict)                    // load the state_dict to model
```
## Contact

For further questions or details, reach out to Ruheng Wang (wangruheng@mail.sdu.edu.cn)
