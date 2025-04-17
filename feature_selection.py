import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from eBoruta import eBoruta
from matplotlib import pyplot as plt
import seaborn as sns


def run_Boruta(X: np.ndarray, y: np.ndarray, model_type=XGBClassifier, model_init_kwargs=dict(n_estimators=20, verbosity=0)):
    boruta = eBoruta().fit(
        X, y, model_type=model_type,
        model_init_kwargs=model_init_kwargs
    )
    return boruta.features_


def plot_imp_history(df_history: pd.DataFrame, accepted: list, top_n: int = 10):
    """
    Funtction that plots Boruta's deduction process 
    """
    plt.figure(figsize=(10, 6))

    # 1. Get top N accepted features based on last step importance
    last_step = df_history['Step'].max()
    last_importance = df_history[df_history['Step'] == last_step]
    accepted_features = last_importance[last_importance['Feature'].isin(accepted)]

    top_features = (
        accepted_features.groupby('Feature')['Importance']
        .mean()
        .sort_values(ascending=False)
        .index
    )

    # 2. Assign vivid colors to top accepted features
    vivid_palette = sns.color_palette("Set1", n_colors=top_n)
    color_map = dict(zip(top_features, vivid_palette))

    # Keep legend handles here
    handles_dict = {}

    # 3. Plot each feature based on its decision at the last step
    for feature in df_history['Feature'].unique():
        data = df_history[df_history['Feature'] == feature]
        last_decision_row = last_importance[last_importance['Feature'] == feature]

        if last_decision_row.empty:
            continue  # Skip if no data for this feature at last step

        decision = last_decision_row['Decision'].values[0]

        if feature in top_features and decision == "Accepted":
            line = sns.lineplot(x='Step', y='Importance', data=data,
                                label=feature, color=color_map[feature])
            handles_dict[feature] = line
        elif decision == "Rejected":
            line = sns.lineplot(x='Step', y='Importance', data=data,
                                color='grey', linewidth=1, alpha=0.5, label="_nolegend_")
            if "Rejected" not in handles_dict:
                handles_dict["Rejected"] = line
        elif decision == "Tentative":
            line = sns.lineplot(x='Step', y='Importance', data=data,
                                color='black', linestyle='--', linewidth=1.2, alpha=0.6, label="_nolegend_")
            if "Tentative" not in handles_dict:
                handles_dict["Tentative"] = line

    # Create custom legend with top features + rejected and tentative lines
    handles = []
    labels = []
    
    # First add the top features
    for feature in top_features[:top_n]:
        if feature in handles_dict:
            handles.append(plt.Line2D([0], [0], color=color_map[feature]))
            labels.append(feature)
    
    # Then add rejected and tentative at the end
    if "Rejected" in handles_dict:
        handles.append(plt.Line2D([0], [0], color='grey', linewidth=1, alpha=0.5))
        labels.append("Rejected")
    
    if "Tentative" in handles_dict:
        handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, alpha=0.6))
        labels.append("Tentative")
    
    plt.legend(handles=handles, labels=labels)
    plt.tight_layout()
    plt.title(f"Top {top_n} Important Features Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Importance")
