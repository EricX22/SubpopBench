python -m subpopbench.scripts.cache_clip_concepts \
  --dataset CheXpertNoFinding --data-dir "$DATA" \
  --split tr --meta artifacts/concept_scores/chexpert_no_finding_attrNo_meta.json \
  --out artifacts/concept_scores/chexpert_no_finding_attrNo_tr.pt \
  --batch-size 128 --num-workers 8 --device cuda --fp16

python -m subpopbench.scripts.cache_clip_concepts \
  --dataset CheXpertNoFinding --data-dir "$DATA" \
  --split va --meta artifacts/concept_scores/chexpert_no_finding_attrNo_meta.json \
  --out artifacts/concept_scores/chexpert_no_finding_attrNo_va.pt \
  --batch-size 128 --num-workers 8 --device cuda --fp16

python -m subpopbench.scripts.cache_clip_concepts \
  --dataset CheXpertNoFinding --data-dir "$DATA" \
  --split te --meta artifacts/concept_scores/chexpert_no_finding_attrNo_meta.json \
  --out artifacts/concept_scores/chexpert_no_finding_attrNo_te.pt \
  --batch-size 128 --num-workers 8 --device cuda --fp16

python -m subpopbench.scripts.cache_stage1_feats \
  --dataset CheXpertNoFinding --data_dir "$DATA" \
  --split va --train_attr no --image_arch resnet_sup_in1k \
  --stage1_ckpt "$STAGE1_CKPT" \
  --out artifacts/feat_cache/chexpert_no_finding_attrNo_feat_va.pt

python -m subpopbench.scripts.cache_stage1_feats \
  --dataset CheXpertNoFinding --data_dir "$DATA" \
  --split te --train_attr no --image_arch resnet_sup_in1k \
  --stage1_ckpt "$STAGE1_CKPT" \
  --out artifacts/feat_cache/chexpert_no_finding_attrNo_feat_te.pt

python -m subpopbench.scripts.fit_residuals_pca \
  --concept_meta artifacts/concept_scores/chexpert_no_finding_attrNo_meta.json \
  --concept_va   artifacts/concept_scores/chexpert_no_finding_attrNo_va.pt \
  --concept_te   artifacts/concept_scores/chexpert_no_finding_attrNo_te.pt \
  --feat_va      artifacts/feat_cache/chexpert_no_finding_attrNo_feat_va.pt \
  --feat_te      artifacts/feat_cache/chexpert_no_finding_attrNo_feat_te.pt \
  --ridge_lam 0.001 --pca_k 64 \
  --out_va   artifacts/resid_cache/chexpert_no_finding_attrNo_resid_va_pca64.pt \
  --out_te   artifacts/resid_cache/chexpert_no_finding_attrNo_resid_te_pca64.pt \
  --out_meta artifacts/resid_cache/chexpert_no_finding_attrNo_resid_meta_pca64.pt
