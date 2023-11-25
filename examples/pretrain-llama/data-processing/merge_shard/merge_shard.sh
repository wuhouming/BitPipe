SAS_TOKEN="sp=racwdli&st=2023-11-25T09:13:18Z&se=2024-04-30T17:13:18Z&spr=https&sv=2022-11-02&sr=c&sig=dcTe10UuDRh1g3Wh1bcYRenmV%2FUqvq%2BKJ7Ufmiw5WfA%3D"
AZJOB_FILE="examples/pretrain-llama/data-processing/merge_shard/template_merge_shard.yaml"
AZ_OUTPUT_FOLDER="https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/merged_shards"
AZ_SUBS="c7209a17-0d9f-41df-8e45-e0172343698d"
AZ_RESOURCE_GROUP="LLM-Test"
AZ_WORKSPACE="Provisioning-Test"
NUM_TOKEN=43486543872

python examples/pretrain-llama/data-processing/merge_shard/merge_shard.py \
--az-subscription "$AZ_SUBS" \
--az-resource-group "$AZ_RESOURCE_GROUP" \
--az-workspace-name "$AZ_WORKSPACE" \
--az-blob-input-folder "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/ar/translated/en2ar_books_corpus_formatted" \
--az-blob-output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sas-token "$SAS_TOKEN" \
--sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_translated_books3_" \
--shard-size $NUM_TOKEN 

python examples/pretrain-llama/data-processing/merge_shard/merge_shard.py \
--az-subscription "$AZ_SUBS" \
--az-resource-group "$AZ_RESOURCE_GROUP" \
--az-workspace-name "$AZ_WORKSPACE" \
--az-blob-input-folder "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/ar/translated/en2ar_c4" \
--az-blob-output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sas-token "$SAS_TOKEN" \
--sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_translated_c4_" \
--shard-size $NUM_TOKEN 

python examples/pretrain-llama/data-processing/merge_shard/merge_shard.py \
--az-subscription "$AZ_SUBS" \
--az-resource-group "$AZ_RESOURCE_GROUP" \
--az-workspace-name "$AZ_WORKSPACE" \
--az-blob-input-folder "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/ar/translated/en2ar_peS2o/arabic_text_only" \
--az-blob-output-folder-path "$AZ_OUTPUT_FOLDER" \
--az-sas-token "$SAS_TOKEN" \
--sample-yaml-job-file "$AZJOB_FILE" \
--prefix-name "ar_translated_peS20_" \
--shard-size $NUM_TOKEN 