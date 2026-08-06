from typing import Dict, Any, Union
from ...shared.model_parametrs_dto import ModelParamsDto, ModelParamsInputDto

def count_pytorch_parameters(model: ModelParamsInputDto) -> Dict[str, int]:
    """
    Counts total and trainable parameters for any PyTorch model.
    """
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    return ModelParamsDto(total_params, trainable_params, total_params - trainable_params, "pytorch", model.model_name) 

def count_tensorflow_parameters(model: ModelParamsInputDto) -> Dict[str, int]:
    """
    Counts total and trainable parameters for any TensorFlow/Keras model.
    """
    total_params = model.model.count_params()
    trainable_params = sum(w.shape.num_elements() for w in model.model.trainable_weights)
    
    return ModelParamsDto(total_params, trainable_params, total_params - trainable_params, "tensorflow", model.model_name) 

def count_scikit_learn_parameters(model: ModelParamsInputDto) -> Dict[str, int]:
    """
    Estimates the parameter count for Scikit-Learn models (e.g., LinearRegression, SVM, Tree-based).
    Returns total weights/coefficients learned during training.
    """
    total_params = 0
    
    # Linear models & Logistic Regression
    if hasattr(model.model, "coef_"):
        total_params += model.model.coef_.size
        if hasattr(model.model, "intercept_"):
            total_params += model.model.intercept_.size
            
    # Support Vector Machines
    elif hasattr(model.model, "support_vectors_"):
        if hasattr(model.model, "dual_coef_"):
            total_params += model.model.dual_coef_.size
        if hasattr(model.model, "intercept_"):
            total_params += model.model.intercept_.size
            
    # Tree-based algorithms (RandomForest, GradientBoosting, DecisionTrees)
    elif hasattr(model.model, "estimators_"):
        for estimator in model.model.estimators_:
            if hasattr(estimator.tree_, "node_count"):
                total_params += estimator.tree_.node_count
    elif hasattr(model.model, "tree_"):
        total_params += model.model.tree_.node_count

    return ModelParamsDto(total_params, total_params, 0, "scikit-learn", model.model_name) 

def calculate_model_parameters(model: ModelParamsInputDto) -> Dict[str, int]:
    """
    Unified entry point. Automatically detects the library/framework of the 
    provided model and returns a standardized payload dictionary.
    """
    model_class_str = str(type(model)).lower()
    
    if "torch" in model_class_str:
        return count_pytorch_parameters(model)
        
    elif "keras" in model_class_str or "tensorflow" in model_class_str:
        return count_tensorflow_parameters(model)
        
    elif "sklearn" in model_class_str:
        return count_scikit_learn_parameters(model)
        
    else:
        raise TypeError(f"Unsupported model framework type: {type(model)}")
