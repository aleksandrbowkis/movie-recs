""" Module for evaluation metrics """

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calculate_metrics(df_predictions, k=10):
    """Calculate RMSE, MAE, Precision@K, Recall@K"""
    y_true = df_predictions['actual_rating']
    y_pred = df_predictions['predicted_rating']

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'precision@k': precision_at_k(df_predictions, k),
        'recall@k': recall_at_k(df_predictions, k)
    }
    return metrics

def precision_at_k(df_predictions, k=10, relevance_score=0.6):
    """
    Calculate Precision@K for all users and return the average. Defined as # of relevant items in top K predictions / K
    
    Args:
        df_predictions: DataFrame with userId, movieId, actual_rating, and predicted_rating
        k: Number of top recommended items to consider (default: 5)
    
    Returns:
        Average Precision@K over all users
    """

    #INitialise list to hold precision@k values
    precision_k_list = []

    for user_id, user_predictions in df_predictions.groupby('userId'):

        # If there are no predictions for this user, skip this user
        if len(user_predictions) == 0:
            continue

        # Sort by predicted rating (descending order)
        user_predictions_sorted = user_predictions.sort_values(by='predicted_rating', ascending=False)

        # Select the top K predicted movies
        top_k_predictions = user_predictions_sorted.head(k).copy()

        # Calculate the number of relevant items in the top K predictions where relevant means actual rating > 3
        top_k_predictions['relevance'] = top_k_predictions['actual_rating'] > relevance_score

        # Calculate Precision@K for this user: # of relevant items in top K / K
        precision_at_k_user = top_k_predictions['relevance'].sum() / k

        precision_k_list.append(precision_at_k_user)
    
    #Average Precision@K over all users
    mean_precision_at_k = np.mean(precision_k_list)
    return mean_precision_at_k

def recall_at_k(df_predictions, k=5, relevance_score=0.6):
    """
    Calculate Recall@K for all users and return the average. 
    Defined as # of relevant items in top K predictions / total # relevant items for user averaged for all users.
    
    Args:
        df_predictions: DataFrame with userId, movieId, actual_rating, and predicted_rating
        k: Number of top recommended items to consider (default: 5)
    
    Returns:
        Average Recall@K over all users
    """

    #INitialise list to hold recall@k values
    recall_k_list = []

    for user_id, user_predictions in df_predictions.groupby('userId'):

        # If there are no predictions for this user, skip this user
        if len(user_predictions) == 0:
            continue

        # Sort by predicted rating (descending order)
        user_predictions_sorted = user_predictions.sort_values(by='predicted_rating', ascending=False)

        # Select the top K predicted movies
        top_k_predictions = user_predictions_sorted.head(k).copy() # Do not modify original dataframe

        # Calculate the number of relevant items in the top K predictions where relevant means actual rating > 3
        top_k_predictions['relevance'] = top_k_predictions['actual_rating'] > relevance_score

        # Calculate total number of relevant items for user
        total_relevant_items = user_predictions[user_predictions['actual_rating'] >= relevance_score]

        # Handle / 0 error if no relevant items
        if len(total_relevant_items) == 0:
            recall_at_k_user = 0
        else:
            # Calculate Recall@K for this user: # of relevant items in top K / total # relevant items for user
            recall_at_k_user = top_k_predictions['relevance'].sum() / len(total_relevant_items)

        recall_k_list.append(recall_at_k_user)
    
    #Average Recall@K over all users.
    mean_recall_at_k = np.mean(recall_k_list)
    
    return mean_recall_at_k