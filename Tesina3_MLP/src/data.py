from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import DataConfig
import pandas as pd

def load_and_preprocess_data(config: DataConfig):
    """
    Loads and preprocesses the dataset.

    - Reading the dataset CSV file
    - Separating features (X) and target (y)
    - Splitting the data into training and test sets
    - Standardizing the feature values
    - Reducing dimensionality via PCA (using percent variance)

    Returns:
        tuple: (X_train_pca, X_test_pca, y_train, y_test)
    """

    # Dataset loading
    df = pd.read_csv(config.dataset_path)

    # Separation of features and targets
    X = df.drop(columns=['ID', 'class'])
    y = (df['class'] == 'P').astype(int)  # 'P' -> 1, 'C' -> 0

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state
    )

    # Data standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # PCA: it uses the percentage of variance 
    pca = PCA(n_components=config.n_components, random_state=config.random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)


    return X_train_pca, X_test_pca, y_train, y_test
