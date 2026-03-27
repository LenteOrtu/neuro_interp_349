import torch
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    FeatureAblation,
    LayerIntegratedGradients,
    LayerGradientXActivation,
    LayerConduction,
    LayerActivations
)

# --- CAPTUM USAGE SNIPPETS ---

# 1. Integrated Gradients
# ig = IntegratedGradients(model)
# attributions, delta = ig.attribute(inputs, target=target_class, return_convergence_delta=True)

# 2. Gradient SHAP
# gs = GradientShap(model)
# attributions = gs.attribute(inputs, baselines=baselines, target=target_class)

# 3. Feature Ablation
# fa = FeatureAblation(model)
# attributions = fa.attribute(inputs, target=target_class)

# 4. Layer Gradient x Activation
# lga = LayerGradientXActivation(model, layer)
# attributions = lga.attribute(inputs, target=target_class)

# 5. Layer Conduction
# lc = LayerConduction(model, layer)
# attributions = lc.attribute(inputs, target=target_class)

# 6. Layer Activations
# la = LayerActivations(model, layer)
# activations = la.attribute(inputs)
