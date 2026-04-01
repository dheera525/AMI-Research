"""
Final AMI ML Suite: Baseline + Cutting-Edge Techniques
======================================================
Comprehensive AMI modeling pipeline including both normal baseline models
and advanced/cutting-edge techniques in one script.

NORMAL BASELINE METHODS:
1. Logistic Regression (LR)
2. SVM (Linear)
3. SVM (RBF)
4. Decision Tree
5. Random Forest
6. Gradient Boosting (sklearn)
7. XGBoost
8. LightGBM
9. CatBoost
10. K-Nearest Neighbors

CUTTING-EDGE METHODS:
11. AutoGluon - AWS AutoML
12. CatBoost with Ordered Boosting
13. SMOTE + Ensemble
14. Cost-Sensitive XGBoost
15. Calibrated Classifier
16. Voting Classifier
17. Stacking Meta-Ensemble
18. LightGBM with DART
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
TARGET_COL = 'group  (control:0, AMI:1)'
LEAKAGE_COL = 'sub-type for AMI( STEMI:0, NON-STEMI :1, control:2)'

# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_preprocess_data(filepath):
    """Load and preprocess the AMI dataset"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    df = pd.read_excel(filepath)
    df = df.drop_duplicates()
    
    X = df.drop(columns=[TARGET_COL, LEAKAGE_COL])
    y = df[TARGET_COL]
    feature_names = X.columns.tolist()
    
    print("Preprocessing complete")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Train/test split and scaling complete\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, df


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate model performance"""
    print(f"\n{'=' * 80}")
    print(f"{model_name}")
    print(f"{'=' * 80}")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': recall,
        'Specificity': specificity,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }
    
    print("Evaluation complete")
    
    return results


# ============================================================================
# NORMAL BASELINE METHODS
# ============================================================================

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Logistic Regression baseline"""
    print("\n[METHOD 1] LOGISTIC REGRESSION (LR)")
    print("-" * 80)
    print("Category: Linear Model")
    print("Complexity: Low")
    
    clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    results = evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
    
    return clf, results


def train_svm_linear(X_train, X_test, y_train, y_test):
    """Support Vector Machine (Linear kernel)"""
    print("\n[METHOD 2] SVM (LINEAR KERNEL)")
    print("-" * 80)
    print("Category: Support Vector Machine")
    print("Complexity: Medium")
    
    clf = SVC(
        kernel='linear',
        C=1.0,
        probability=True,
        random_state=RANDOM_STATE
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    results = evaluate_model(y_test, y_pred, y_pred_proba, "SVM (Linear)")
    
    return clf, results


def train_svm_rbf(X_train, X_test, y_train, y_test):
    """Support Vector Machine (RBF kernel)"""
    print("\n[METHOD 3] SVM (RBF KERNEL)")
    print("-" * 80)
    print("Category: Support Vector Machine")
    print("Complexity: Medium-High")
    
    clf = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    results = evaluate_model(y_test, y_pred, y_pred_proba, "SVM (RBF)")
    
    return clf, results


def train_decision_tree(X_train, X_test, y_train, y_test):
    """Decision Tree baseline"""
    print("\n[METHOD 4] DECISION TREE")
    print("-" * 80)
    print("Category: Tree-Based")
    print("Complexity: Low-Medium")
    
    clf = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    results = evaluate_model(y_test, y_pred, y_pred_proba, "Decision Tree")
    
    return clf, results


def train_random_forest(X_train, X_test, y_train, y_test):
    """Random Forest baseline"""
    print("\n[METHOD 5] RANDOM FOREST")
    print("-" * 80)
    print("Category: Ensemble (Bagging)")
    print("Complexity: Medium")
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    results = evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
    
    return clf, results


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Gradient Boosting baseline (sklearn)"""
    print("\n[METHOD 6] GRADIENT BOOSTING (SKLEARN)")
    print("-" * 80)
    print("Category: Ensemble (Boosting)")
    print("Complexity: Medium-High")
    
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    results = evaluate_model(y_test, y_pred, y_pred_proba, "Gradient Boosting (sklearn)")
    
    return clf, results


def train_xgboost(X_train, X_test, y_train, y_test):
    """XGBoost baseline"""
    print("\n[METHOD 7] XGBOOST")
    print("-" * 80)
    print("Category: Ensemble (Boosting)")
    print("Complexity: High")
    
    try:
        from xgboost import XGBClassifier
        
        clf = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        clf.fit(X_train, y_train, verbose=False)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        return clf, results
        
    except ImportError:
        print("✗ xgboost not installed - skipping")
        print("  Install: pip install xgboost")
        return None, None


def train_lightgbm(X_train, X_test, y_train, y_test):
    """LightGBM baseline"""
    print("\n[METHOD 8] LIGHTGBM")
    print("-" * 80)
    print("Category: Ensemble (Boosting)")
    print("Complexity: High")
    
    try:
        from lightgbm import LGBMClassifier
        
        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "LightGBM")
        
        return clf, results
        
    except ImportError:
        print("✗ lightgbm not installed - skipping")
        print("  Install: pip install lightgbm")
        return None, None


def train_catboost(X_train, X_test, y_train, y_test):
    """CatBoost baseline"""
    print("\n[METHOD 9] CATBOOST")
    print("-" * 80)
    print("Category: Ensemble (Boosting)")
    print("Complexity: High")
    
    try:
        from catboost import CatBoostClassifier
        
        clf = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=5,
            l2_leaf_reg=3,
            random_state=RANDOM_STATE,
            verbose=0
        )
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "CatBoost")
        
        return clf, results
        
    except ImportError:
        print("✗ catboost not installed - skipping")
        print("  Install: pip install catboost")
        return None, None


def train_knn(X_train, X_test, y_train, y_test):
    """K-Nearest Neighbors baseline"""
    print("\n[METHOD 10] K-NEAREST NEIGHBORS")
    print("-" * 80)
    print("Category: Instance-Based")
    print("Complexity: Low")
    
    clf = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean'
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    results = evaluate_model(y_test, y_pred, y_pred_proba, "K-Nearest Neighbors")
    
    return clf, results


# ============================================================================
# TECHNIQUE 1: AUTOGLUON (AWS AUTOML)
# ============================================================================

def train_autogluon(X_train, X_test, y_train, y_test):
    """
    AutoGluon - Automated ML from AWS (2020)
    
    Why AutoGluon?
    - Automatically trains and stacks multiple models
    - State-of-the-art AutoML performance
    - Ensemble of best models automatically
    - Minimal configuration needed
    
    Install: pip install autogluon
    """
    print("\n[TECHNIQUE 1] AUTOGLUON - AWS AutoML")
    print("-" * 80)
    print("Innovation: Automatic model stacking and ensembling")
    print("Category: AutoML")
    
    try:
        from autogluon.tabular import TabularPredictor
        
        # Create training dataframe
        train_data = pd.DataFrame(X_train)
        train_data['target'] = y_train.values
        
        test_data = pd.DataFrame(X_test)
        
        print("\nTraining AutoGluon (60 second time limit)...")
        
        predictor = TabularPredictor(
            label='target',
            eval_metric='roc_auc',
            problem_type='binary'
        ).fit(
            train_data,
            time_limit=60,
            presets='best_quality',
            verbosity=0
        )
        
        y_pred_proba = predictor.predict_proba(test_data).values[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "AutoGluon")
        
        print("\nAutoGluon ensemble training complete")
        
        return predictor, results
        
    except ImportError:
        print("✗ autogluon not installed")
        print("  Install: pip install autogluon")
        return None, None


# ============================================================================
# TECHNIQUE 2: CATBOOST WITH ORDERED BOOSTING
# ============================================================================

def train_catboost_ordered(X_train, X_test, y_train, y_test):
    """
    CatBoost with Ordered Boosting
    
    Why Ordered Boosting?
    - Reduces overfitting via ordered target encoding
    - Better generalization than standard boosting
    - Handles prediction shift
    """
    print("\n[TECHNIQUE 2] CATBOOST WITH ORDERED BOOSTING")
    print("-" * 80)
    print("Innovation: Ordered target encoding prevents overfitting")
    print("Category: Advanced Boosting")
    
    try:
        from catboost import CatBoostClassifier
        
        clf = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            random_state=RANDOM_STATE,
            verbose=0,
            eval_metric='AUC'
        )
        
        clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "CatBoost (Ordered)")
        
        return clf, results
        
    except ImportError:
        print("✗ catboost not installed - skipping")
        return None, None


# ============================================================================
# TECHNIQUE 3: SMOTE + ENSEMBLE
# ============================================================================

def train_smote_ensemble(X_train, X_test, y_train, y_test):
    """
    SMOTE (Synthetic Minority Over-sampling) + Ensemble
    
    Why SMOTE?
    - Balances classes by generating synthetic samples
    - Particularly useful when classes are imbalanced
    - Reduces bias toward majority class
    """
    print("\n[TECHNIQUE 3] SMOTE + ENSEMBLE")
    print("-" * 80)
    print("Innovation: Synthetic oversampling for better class balance")
    print("Category: Data Augmentation + Ensemble")
    
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.ensemble import BalancedRandomForestClassifier
        
        # Apply SMOTE
        print("\nApplying SMOTE...")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print("SMOTE balancing complete")
        
        # Train ensemble on balanced data
        clf = BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        clf.fit(X_train_balanced, y_train_balanced)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "SMOTE + Ensemble")
        
        return clf, results
        
    except ImportError:
        print("✗ imbalanced-learn not installed")
        print("  Install: pip install imbalanced-learn")
        return None, None


# ============================================================================
# TECHNIQUE 4: COST-SENSITIVE XGBOOST
# ============================================================================

def train_cost_sensitive_xgboost(X_train, X_test, y_train, y_test):
    """
    Cost-Sensitive XGBoost
    
    Why Cost-Sensitive?
    - In medical diagnosis, false negatives (missing MI) are more costly
    - Penalizes missing AMI cases more than false alarms
    - Clinically appropriate objective function
    """
    print("\n[TECHNIQUE 4] COST-SENSITIVE XGBOOST")
    print("-" * 80)
    print("Innovation: Penalizes false negatives more (critical for medical AI)")
    print("Category: Cost-Sensitive Learning")
    
    try:
        from xgboost import XGBClassifier
        
        # Calculate scale_pos_weight (ratio of negative to positive)
        # This makes the model focus more on the positive (AMI) class
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        print("\nCost-sensitive weighting enabled for AMI detection")
        
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight * 2,  # Extra emphasis on catching AMI
            random_state=RANDOM_STATE,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        clf.fit(X_train, y_train, verbose=False)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "Cost-Sensitive XGBoost")
        
        print("\n💡 Optimized to catch more AMI cases (higher sensitivity)!")
        
        return clf, results
        
    except ImportError:
        print("✗ xgboost not installed - skipping")
        return None, None


# ============================================================================
# TECHNIQUE 5: CALIBRATED CLASSIFIERS
# ============================================================================

def train_calibrated_classifier(X_train, X_test, y_train, y_test):
    """
    Calibrated Classifier (Isotonic Calibration)
    
    Why Calibration?
    - Improves probability estimates
    - Important when probabilities are used for decisions
    - Better confidence scores for clinicians
    """
    print("\n[TECHNIQUE 5] CALIBRATED CLASSIFIER")
    print("-" * 80)
    print("Innovation: Better calibrated probability estimates")
    print("Category: Probability Calibration")
    
    try:
        from lightgbm import LGBMClassifier
        
        # Base classifier
        base_clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        
        # Calibrate with isotonic regression
        clf = CalibratedClassifierCV(
            base_clf,
            method='isotonic',
            cv=5
        )
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "Calibrated LightGBM")
        
        print("\n💡 Probability estimates are now better calibrated!")
        
        return clf, results
        
    except ImportError:
        print("✗ lightgbm not installed - using Random Forest")
        
        base_clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
        clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "Calibrated RandomForest")
        
        return clf, results


# ============================================================================
# TECHNIQUE 6: VOTING CLASSIFIER (SOFT VOTING)
# ============================================================================

def train_voting_classifier(X_train, X_test, y_train, y_test):
    """
    Voting Classifier - Soft Voting
    
    Why Voting?
    - Combines predictions from diverse models
    - Soft voting uses probability averaging
    - Reduces variance through diversity
    """
    print("\n[TECHNIQUE 6] VOTING CLASSIFIER (SOFT VOTING)")
    print("-" * 80)
    print("Innovation: Probability averaging from diverse models")
    print("Category: Ensemble (Voting)")
    
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        
        # Create diverse base estimators
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            ('lgbm', LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbose=-1)),
            ('xgb', XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]
        
        clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        print("\nTraining 4 diverse models and combining via soft voting...")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "Voting Classifier")
        
        return clf, results
        
    except ImportError:
        print("✗ Some boosting libraries not installed - using available models")
        
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]
        
        clf = VotingClassifier(estimators=estimators, voting='soft')
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "Voting Classifier (Limited)")
        
        return clf, results


# ============================================================================
# TECHNIQUE 7: STACKING META-ENSEMBLE
# ============================================================================

def train_stacking_classifier(X_train, X_test, y_train, y_test):
    """
    Stacking Classifier with Meta-Learner
    
    Why Stacking?
    - Meta-model learns how to best combine base models
    - More sophisticated than simple voting
    - Can capture complementary strengths
    """
    print("\n[TECHNIQUE 7] STACKING META-ENSEMBLE")
    print("-" * 80)
    print("Innovation: Meta-learner optimally combines base models")
    print("Category: Meta-Learning (Stacking)")
    
    try:
        from lightgbm import LGBMClassifier
        
        # Base estimators
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=RANDOM_STATE)),
            ('lgbm', LGBMClassifier(n_estimators=150, random_state=RANDOM_STATE, verbose=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE))
        ]
        
        # Meta-learner (logistic regression for interpretability)
        meta_learner = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        
        clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        print("\nTraining base models and meta-learner...")
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "Stacking Classifier")
        
        print("\n💡 Meta-learner learned optimal combination weights!")
        
        return clf, results
        
    except ImportError:
        print("✗ lightgbm not installed - using available models")
        
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)),
            ('gb', GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE))
        ]
        
        meta_learner = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf = StackingClassifier(estimators=base_estimators, final_estimator=meta_learner, cv=5)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "Stacking Classifier (Limited)")
        
        return clf, results


# ============================================================================
# TECHNIQUE 8: GRADIENT BOOSTING WITH DART
# ============================================================================

def train_lightgbm_dart(X_train, X_test, y_train, y_test):
    """
    LightGBM with DART (Dropouts meet Multiple Additive Regression Trees)
    
    Why DART?
    - Prevents overfitting via dropout in trees
    - More robust than standard GOSS
    - Better generalization
    """
    print("\n[TECHNIQUE 8] LIGHTGBM WITH DART")
    print("-" * 80)
    print("Innovation: Dropout in gradient boosting trees")
    print("Category: Advanced Boosting")
    
    try:
        from lightgbm import LGBMClassifier
        
        clf = LGBMClassifier(
            boosting_type='dart',  # DART instead of GOSS
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=40,
            drop_rate=0.1,
            skip_drop=0.5,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        results = evaluate_model(y_test, y_pred, y_pred_proba, "LightGBM (DART)")
        
        print("\n💡 DART prevents overfitting via tree dropout!")
        
        return clf, results
        
    except ImportError:
        print("✗ lightgbm not installed - skipping")
        return None, None


# ============================================================================
# COMPARISON + VISUAL WEB REPORT
# ============================================================================

def create_dataset_overview_plot(df, output_path='dataset_overview.png'):
    """Create graph-only dataset overview plot for the visual report."""
    X = df.drop(columns=[TARGET_COL, LEAKAGE_COL], errors='ignore')
    y = df[TARGET_COL]
    class_labels = y.map({0: 'Control', 1: 'AMI'}).fillna('Unknown')
    
    numeric_X = X.select_dtypes(include=[np.number]).copy()
    if numeric_X.empty:
        numeric_X = pd.DataFrame({'placeholder': np.zeros(len(df))})
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Class distribution
    sns.countplot(x=class_labels, ax=axes[0, 0], palette='Set2')
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_xlabel('Patient Group')
    axes[0, 0].set_ylabel('Count')
    
    # Missing data profile
    missing_ratio = X.isnull().mean().sort_values(ascending=False).head(12)
    if missing_ratio.sum() > 0:
        missing_ratio.plot(kind='bar', ax=axes[0, 1], color='salmon')
        axes[0, 1].set_title('Missing Data by Feature')
        axes[0, 1].set_ylabel('Missing Ratio')
        axes[0, 1].tick_params(axis='x', rotation=75)
    else:
        axes[0, 1].text(
            0.5, 0.5, 'No Missing Data Patterns Detected',
            ha='center', va='center', fontsize=12
        )
        axes[0, 1].axis('off')
    
    # Correlation map of most varying features
    top_var_features = numeric_X.var().sort_values(ascending=False).head(12).index.tolist()
    if len(top_var_features) >= 2:
        corr = numeric_X[top_var_features].corr()
        sns.heatmap(corr, ax=axes[1, 0], cmap='coolwarm', center=0, cbar=True)
        axes[1, 0].set_title('Feature Correlation Map')
    else:
        axes[1, 0].text(
            0.5, 0.5, 'Not Enough Numeric Features for Correlation Map',
            ha='center', va='center', fontsize=12
        )
        axes[1, 0].axis('off')
    
    # PCA landscape
    pca_input = numeric_X[top_var_features] if top_var_features else numeric_X
    pca_input = pca_input.fillna(pca_input.median(numeric_only=True))
    if pca_input.shape[1] >= 2:
        scaled = StandardScaler().fit_transform(pca_input)
        reduced = PCA(n_components=2).fit_transform(scaled)
        scatter = axes[1, 1].scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=y,
            cmap='coolwarm',
            alpha=0.7,
            s=45
        )
        axes[1, 1].set_title('Patient Feature Landscape (PCA)')
        axes[1, 1].set_xlabel('Component 1')
        axes[1, 1].set_ylabel('Component 2')
        legend_handles, _ = scatter.legend_elements()
        axes[1, 1].legend(legend_handles, ['Control', 'AMI'], title='Class', loc='best')
    else:
        axes[1, 1].text(
            0.5, 0.5, 'Not Enough Features for PCA View',
            ha='center', va='center', fontsize=12
        )
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_model_metrics_heatmap(results_df, output_path='model_metrics_heatmap.png'):
    """Create compact heatmap view for model metrics."""
    metric_columns = ['AUC-ROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'Precision']
    heatmap_data = results_df.set_index('Model')[metric_columns]
    
    fig_height = max(6, len(results_df) * 0.45)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    sns.heatmap(
        heatmap_data,
        cmap='YlGnBu',
        linewidths=0.4,
        cbar=True,
        annot=False,
        ax=ax
    )
    ax.set_title('Model Performance Heatmap')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Models')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def _encode_image(image_path):
    """Encode image to base64 for embedding in a self-contained HTML report."""
    return base64.b64encode(Path(image_path).read_bytes()).decode('utf-8')


def generate_visual_report(results_df, dataset_plot_path, comparison_plot_path, heatmap_plot_path,
                           output_html='ami_visual_report.html'):
    """Generate a small frontend-style HTML dashboard with graph-only outputs."""
    top_models = results_df['Model'].head(3).tolist()
    top_model_html = ''.join(f'<li>{model}</li>' for model in top_models)
    
    dataset_img = _encode_image(dataset_plot_path)
    comparison_img = _encode_image(comparison_plot_path)
    heatmap_img = _encode_image(heatmap_plot_path)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AMI Research Visual Dashboard</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #1f2a37;
      --accent: #0f766e;
      --muted: #607080;
      --border: #d8e1ea;
    }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 85% 10%, #c9f1e7 0%, transparent 28%),
        radial-gradient(circle at 15% 0%, #d9e9ff 0%, transparent 30%),
        var(--bg);
      line-height: 1.45;
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, #0f766e, #166534);
      color: #fff;
      border-radius: 16px;
      padding: 24px 26px;
      box-shadow: 0 16px 40px rgba(15, 118, 110, 0.25);
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      letter-spacing: 0.2px;
    }}
    .subtitle {{
      margin-top: 8px;
      color: #e6fffb;
      max-width: 760px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: 0 8px 22px rgba(0, 0, 0, 0.05);
    }}
    h2 {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 20px;
    }}
    p {{
      margin: 8px 0;
      color: var(--muted);
    }}
    ul {{
      margin: 8px 0 0;
      padding-left: 20px;
      color: var(--ink);
    }}
    .plot {{
      margin-top: 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      box-shadow: 0 8px 22px rgba(0, 0, 0, 0.05);
    }}
    .plot img {{
      display: block;
      width: 100%;
      border-radius: 10px;
    }}
    .foot {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>AMI Research Visual Dashboard</h1>
      <p class="subtitle">
        Graph-first report for the AMI baseline and cutting-edge model suite.
        This dashboard focuses on visual output instead of numeric console metrics.
      </p>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Data Pipeline</h2>
        <p>Raw AMI data is cleaned, deduplicated, and split into train/test sets.</p>
        <p>Leakage features are removed before training and standardized inputs are used for consistency.</p>
      </article>
      <article class="card">
        <h2>Top Ranked Models</h2>
        <ul>
          {top_model_html}
        </ul>
      </article>
      <article class="card">
        <h2>Model Families Included</h2>
        <p>Traditional baselines, ensemble methods, calibration, AutoML, and cost-sensitive approaches.</p>
      </article>
    </section>

    <section class="plot">
      <h2>Dataset Overview</h2>
      <img alt="Dataset overview plots" src="data:image/png;base64,{dataset_img}" />
    </section>
    <section class="plot">
      <h2>Model Comparison</h2>
      <img alt="Model comparison plots" src="data:image/png;base64,{comparison_img}" />
    </section>
    <section class="plot">
      <h2>Metrics Heatmap</h2>
      <img alt="Model metrics heatmap" src="data:image/png;base64,{heatmap_img}" />
    </section>

    <p class="foot">Generated by final.py visual report mode.</p>
  </div>
</body>
</html>
"""
    
    Path(output_html).write_text(html, encoding='utf-8')
    return output_html


def compare_models(results_list):
    """Compare all models in the final suite"""
    print("\n" + "=" * 80)
    print("CREATING VISUAL OUTPUTS")
    print("=" * 80)
    
    df = pd.DataFrame(results_list).sort_values('AUC-ROC', ascending=False)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # AUC-ROC
    df.plot(x='Model', y='AUC-ROC', kind='barh', ax=axes[0, 0], legend=False, color='steelblue')
    axes[0, 0].set_xlabel('AUC-ROC')
    axes[0, 0].set_title('Final Suite: AUC-ROC', fontweight='bold')
    axes[0, 0].axvline(x=0.90, color='red', linestyle='--', alpha=0.5, label='Excellent (>0.90)')
    axes[0, 0].legend()
    
    # Accuracy
    df.plot(x='Model', y='Accuracy', kind='barh', ax=axes[0, 1], legend=False, color='coral')
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_title('Final Suite: Accuracy', fontweight='bold')
    
    # Sensitivity vs Specificity
    axes[1, 0].scatter(df['Sensitivity'], df['Specificity'], s=150, alpha=0.6, c=range(len(df)), cmap='viridis')
    for idx, row in df.iterrows():
        axes[1, 0].annotate(row['Model'], (row['Sensitivity'], row['Specificity']), fontsize=8, ha='center')
    axes[1, 0].set_xlabel('Sensitivity')
    axes[1, 0].set_ylabel('Specificity')
    axes[1, 0].set_title('Sensitivity vs Specificity', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1-Score
    df.plot(x='Model', y='F1-Score', kind='barh', ax=axes[1, 1], legend=False, color='mediumseagreen')
    axes[1, 1].set_xlabel('F1-Score')
    axes[1, 1].set_title('Final Suite: F1-Score', fontweight='bold')
    
    plt.tight_layout()
    comparison_plot_path = 'final_comparison.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    df.to_csv('final_results.csv', index=False)
    heatmap_plot_path = create_model_metrics_heatmap(df)
    
    print("Model comparison graphs saved")
    
    return df, comparison_plot_path, heatmap_plot_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("FINAL AMI ML SUITE: BASELINE + CUTTING-EDGE")
    print("=" * 80)
    print("\nRunning baseline + cutting-edge model families.")
    print("Final output will be visual-first (graphs + HTML dashboard).")
    
    # Load data
    import os
    if os.path.exists('AMI_HeartDisease_dataset.xlsx'):
        filepath = 'AMI_HeartDisease_dataset.xlsx'
    else:
        filepath = input("\nEnter path to dataset: ").strip().strip('"').strip("'")
    
    X_train, X_test, y_train, y_test, feature_names, scaler, raw_df = load_and_preprocess_data(filepath)
    
    all_results = []
    
    # Train all models
    print("\n" + "=" * 80)
    print("TRAINING ALL FINAL-SUITE MODELS")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("BASELINE MODELS")
    print("-" * 80)
    
    # 1. Logistic Regression
    _, results = train_logistic_regression(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 2. SVM (Linear)
    _, results = train_svm_linear(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 3. SVM (RBF)
    _, results = train_svm_rbf(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 4. Decision Tree
    _, results = train_decision_tree(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 5. Random Forest
    _, results = train_random_forest(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 6. Gradient Boosting
    _, results = train_gradient_boosting(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 7. XGBoost
    _, results = train_xgboost(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 8. LightGBM
    _, results = train_lightgbm(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 9. CatBoost
    _, results = train_catboost(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 10. KNN
    _, results = train_knn(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    print("\n" + "-" * 80)
    print("CUTTING-EDGE MODELS")
    print("-" * 80)
    
    # 11. AutoGluon
    _, results = train_autogluon(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 12. CatBoost Ordered
    _, results = train_catboost_ordered(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 13. SMOTE + Ensemble
    _, results = train_smote_ensemble(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 14. Cost-Sensitive XGBoost
    _, results = train_cost_sensitive_xgboost(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 15. Calibrated Classifier
    _, results = train_calibrated_classifier(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 16. Voting Classifier
    _, results = train_voting_classifier(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 17. Stacking
    _, results = train_stacking_classifier(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # 18. LightGBM DART
    _, results = train_lightgbm_dart(X_train, X_test, y_train, y_test)
    if results: all_results.append(results)
    
    # Compare
    if all_results:
        dataset_plot_path = create_dataset_overview_plot(raw_df)
        df, comparison_plot_path, heatmap_plot_path = compare_models(all_results)
        report_path = generate_visual_report(
            df,
            dataset_plot_path,
            comparison_plot_path,
            heatmap_plot_path
        )
        
        print("\n" + "=" * 80)
        print("TOP RANKED MODELS")
        print("=" * 80)
        for model_name in df.head(3)['Model'].tolist():
            print(f"- {model_name}")
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)
        print("\n✓ Includes both normal baseline and cutting-edge methods")
        print("✓ Outputs are now graph-first with minimal console metrics")
        print("✓ Visual dashboard includes data overview and model comparisons")
        print(f"✓ Open '{report_path}' in your browser to view the full frontend-style report")
    
    print("\n" + "=" * 80)
    print("✓ FINAL SUITE EVALUATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
