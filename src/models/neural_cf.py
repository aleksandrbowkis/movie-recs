# Defines class MovieRecommender for neural collaborative filtering model
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from typing import Tuple, List

class MovieRecommender:
    def __init__(
        self, 
        n_users: int, 
        n_movies: int, 
        embedding_dim: int = 50,
        hidden_layers: List[int] = [128],
        dropout_rate: float = 0.2
    ):
        """
        Neural Collaborative Filtering model for movie recommendations.
        
        Args:
            n_users: Number of unique users
            n_movies: Number of unique movies
            embedding_dim: Dimension of embedding vectors
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        # Input layers
        user_input = Input(shape=(1,), name="user_input")
        movie_input = Input(shape=(1,), name="movie_input")
        
        # Embedding layers
        user_embedding = Embedding(
            input_dim=self.n_users,
            output_dim=self.embedding_dim,
            name="user_embedding"
        )(user_input)
        movie_embedding = Embedding(
            input_dim=self.n_movies,
            output_dim=self.embedding_dim,
            name="movie_embedding"
        )(movie_input)
        
        # Flatten embeddings
        user_vec = Flatten()(user_embedding)
        movie_vec = Flatten()(movie_embedding)
        
        # Combine embeddings
        concat = Concatenate()([user_vec, movie_vec])
        
        # Hidden layers. Dropout used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
        # Takes embeddings as first input and passes through hidden layers, each time randomly dropping out a fraction of units.
        x = concat
        for units in self.hidden_layers:
            x = Dense(units, activation="relu")(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            
        # Output layer
        # Predict rating. Single neuron as predicting a single score for a user/movie pair
        # Use sigmoid activation function as ratings are bounded [0,1]  
        output = Dense(1, activation="sigmoid")(x)
        
        # Build model
        model = Model(inputs=[user_input, movie_input], outputs=output)
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with specified optimizer and metrics."""
        # Learning rate defines the step size taken during optimisation
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae", "mse"]
        )
    
    def train(
        self,
        X_train: Tuple[np.ndarray, np.ndarray],
        y_train: np.ndarray,
        validation_split: float = 0.2,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 3
    ):
        """
        Train the model with early stopping.
        
        Args:
            X_train: Tuple of (user_ids, movie_ids) input data
            y_train: Target ratings
            validation_split: Fraction of data to use for validation
            batch_size: Training batch size
            epochs: Maximum number of epochs
            patience: Early stopping patience
        
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        history = self.model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test: Tuple[np.ndarray, np.ndarray], y_test: np.ndarray):
        """Evaluate model performance on test data."""
        metrics = self.model.evaluate(X_test, y_test, verbose=0)
        return dict(zip(self.model.metrics_names, metrics))
    
    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Generate predictions for user-movie pairs."""
        return self.model.predict([user_ids, movie_ids], verbose=0)