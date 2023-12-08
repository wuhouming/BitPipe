
python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-allam/training/allam-7bv3-0/data.json" \
 --domain-ratio-from-json  "examples/pretrain-allam/training/allam-7bv3-0/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-allam/training/allam-7bv3-0/lang_prob.json" \
 --exclude-iterator-json "examples/pretrain-allam/training/allam-7bv3-0/exclude_iterator.json" \
 --total-token 300000000000 \
 --export-script examples/pretrain-allam/training/allam-7bv3-0/iterator_prob.sh \
 --prefix-for-file-path "\$BIN_IDX_PATH/"
