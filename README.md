# Knowledge Extraction with open-source LLMs (KNEON)

## Directory Structure
 * data_analysis: Encompasses python scripts for data analysis and understanding for all datasets
 * datasets: Encompasses train and dev json for all datasets in three directories (GermanQuAD, SQuAD1.1, SQuAD2.0)
 * experiments: see below
 * final_paper: pdf and word
* final_presentation: pdf and pptx

###  Experiments

Experiments are in three directories (GermanQuAD, SQuAD1.1, SQuAD2.0).
For SQuAD1.1 and SQuAD2.0, they are further categorized in `bert_family` and `bert_base_cased`. The Jupyter Notebook file have naming convention is `<dataset>_<model_group>_<peft_label>_<hpo_label>`. 
* dataset: germanquad, squad_1_1, squad_2_0
* model_group: german_finetuned_models, bert_family, bert_base_cased
* peft_label: baseline, lora, qlora
* hpo_label: raw, raw_and_hpo (hyperparameter optimization)
