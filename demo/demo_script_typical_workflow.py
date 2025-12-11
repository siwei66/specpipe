# -*- coding: utf-8 -*-
"""
SpecPipe demonstration - Typical workflow for spectral image analysis

Copyright (c) 2025 Siwei Luo. MIT License.
"""

# Setup: create directory and download demonstration data
import os
demo_dir = os.getcwd() + '/specpipe_demo/'
os.makedirs(demo_dir)
from specpipe import download_demo_data
download_demo_data(demo_dir)

# Configure SpecExp
from specpipe import SpecExp
exp = SpecExp(demo_dir + 'reports/')
exp.add_groups(['group_1', 'group_2'])
exp.add_images_by_name('demo.', demo_dir, 'group_1')
exp.add_images_by_name('demo.', demo_dir, 'group_2')
exp.add_rois_by_suffix('_[12].xml', demo_dir, 'group_1')
exp.add_rois_by_suffix('_[345].xml', demo_dir, 'group_2')

# Add labels
labels = exp.ls_labels()  # Get default labels
labels.iloc[:, 1] = exp.ls_rois_sample(return_dataframe=True)["ROI_name"]
exp.sample_labels = labels  # Set new labels
# Add targets
targets = exp.ls_sample_targets()  # Get sample target dataframe
targets["Target_value"] = [f"leaf_{labl[0]}" for labl in targets['Label']]
exp.sample_targets_from_df(targets)  # Set new targets

# Create pipeline
from specpipe import SpecPipe
pipe = SpecPipe(exp)

# Add baseline correction methods
from specpipe.functions import snv
def raw(v): return v  # type: ignore
pipe.add_process(2, 2, 0, raw)  # noqa
pipe.add_process(2, 2, 0, snv)

# Add ROI statistics for modeling data
from specpipe import roi_mean, roi_median
pipe.add_process(5, 7, 0, roi_mean)  # 5 – image ROI to 7 – sample spectra
pipe.add_process(5, 7, 0, roi_median)

# Denoising
from specpipe.denoiser import LocalPolynomial
pipe.add_process(7, 7, 0, raw)  # passthrough
pipe.add_process(7, 7, 0, LocalPolynomial(5, polynomial_order=2).savitzky_golay_filter)

# Create models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
rf = RandomForestClassifier(n_estimators=10)
knn = KNeighborsClassifier(n_neighbors=3)

# Combine feature selectors
from sklearn.feature_selection import SelectKBest, f_classif
from specpipe.modelconnector import IdentityTransformer
from specpipe import factorial_transformer_chains
models = factorial_transformer_chains(
    [SelectKBest(f_classif, k=5), IdentityTransformer()],
    estimators=[knn, rf], is_regression=False)

# Add models
for model in models: pipe.add_model(model, validation_method="2-fold")  # noqa

# Review the processing chains and run pipeline
pipe.ls_chains()
pipe.run()
