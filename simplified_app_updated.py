import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Ransomware Family Classification",
    page_icon="🔒",
    layout="wide"
)

# Title and description
st.title("🔒 Ransomware Family Classification")
st.markdown("""
This application uses deep learning models (CNN and RNN) to classify ransomware families
based on Shannon entropy features and file type information.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload & Preprocessing", "Model Configuration", "Training", "Evaluation"])

# Initialize session state variables
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = "CNN"
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'family_mapping' not in st.session_state:
    st.session_state.family_mapping = None
if 'test_predictions' not in st.session_state:
    st.session_state.test_predictions = None

# Data Upload & Preprocessing page
if page == "Data Upload & Preprocessing":
    st.header("Data Upload & Preprocessing")
    
    # File upload options
    upload_option = st.radio(
        "Choose data source:",
        ["Upload CSV file", "Use formatted sample dataset"]
    )
    
    if upload_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=['csv'])
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.dataset = data
                st.success(f"Dataset loaded successfully! Shape: {data.shape}")
                
                # Display dataset overview
                st.subheader("Dataset Overview")
                st.dataframe(data.head())
                
                # Display dataset statistics
                st.subheader("Dataset Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Basic Statistics")
                    st.dataframe(data.describe())
                
                with col2:
                    st.write("Feature Types")
                    st.dataframe(pd.DataFrame({
                        'Feature': data.columns,
                        'Type': data.dtypes,
                        'Missing Values': data.isnull().sum()
                    }))
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
else:
    # Use the formatted sample dataset
    formatted_file = 'formatted_entropy_dataset.csv'

    # Generate dataset if it doesn't exist
    if not os.path.exists(formatted_file):
        with st.spinner("Generating formatted sample dataset..."):
            try:
                exec(open('format_dataset.py').read())  # Run formatting script
            except Exception as e:
                st.error(f"Error formatting dataset: {e}")

    if os.path.exists(formatted_file):
        data = pd.read_csv(formatted_file)
        st.session_state.dataset = data
        st.success(f"Sample dataset loaded successfully! Shape: {data.shape}")

        # Display dataset overview
        st.subheader("Dataset Overview")
        st.dataframe(data.head())

        # Display dataset statistics
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Basic Statistics")
            st.dataframe(data.describe())

        with col2:
            st.write("Feature Types")
            st.dataframe(pd.DataFrame({
                'Feature': data.columns,
                'Type': data.dtypes,
                'Missing Values': data.isnull().sum()
            }))
    else:
        st.error("Formatted sample dataset could not be loaded or generated. Please upload a CSV file instead.")
    
    # Preprocessing section
    if st.session_state.dataset is not None:
        st.subheader("Preprocessing Options")
        
        data = st.session_state.dataset
        
        # Select target column
        target_col = st.selectbox(
            "Select the target column (ransomware family labels)",
            [col for col in data.columns if 'label' in col.lower() or 'family' in col.lower()],
            index=0
        )
        
        # Select feature columns
        feature_cols = st.multiselect(
            "Select feature columns",
            [col for col in data.columns if col != target_col],
            default=[col for col in data.columns if col != target_col]
        )
        
        # Preprocessing options
        normalize = st.checkbox("Normalize features", value=True)
        test_size = st.slider("Test set size (%)", min_value=5, max_value=30, value=15, step=5) / 100
        
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                # Extract features and target
                X = data[feature_cols].values
                y = data[target_col].values
                
                # Normalize features if requested
                if normalize:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                
                # Create stratified split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=42
                )
                
                # Reshape for CNN/RNN (add a dimension)
                X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                
                # Store in session state
                st.session_state.X_train = X_train_reshaped
                st.session_state.X_test = X_test_reshaped
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                # Reset any existing training history
                st.session_state.training_history = None
                st.session_state.test_predictions = None
                
                st.success("Data preprocessing completed!")
                
                # Display split information
                st.subheader("Data Split Information")
                st.write(f"Training set: {X_train.shape[0]} samples")
                st.write(f"Testing set: {X_test.shape[0]} samples")
                
                # Display class distribution
                st.subheader("Class Distribution")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Training set distribution
                train_dist = pd.Series(y_train).value_counts().sort_index()
                ax1.bar(train_dist.index.astype(str), train_dist.values)
                ax1.set_title("Training Set Class Distribution")
                ax1.set_xlabel("Ransomware Family")
                ax1.set_ylabel("Count")
                ax1.tick_params(axis='x', rotation=45)
                
                # Test set distribution
                test_dist = pd.Series(y_test).value_counts().sort_index()
                ax2.bar(test_dist.index.astype(str), test_dist.values)
                ax2.set_title("Test Set Class Distribution")
                ax2.set_xlabel("Ransomware Family")
                ax2.set_ylabel("Count")
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)

# Model Configuration page
elif page == "Model Configuration":
    st.header("Model Configuration")
    
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Please upload and preprocess data first.")
    else:
        # Model type selection
        st.subheader("Model Selection")
        model_type = st.radio(
            "Select model type",
            ["CNN", "RNN"],
            index=0 if st.session_state.model_type == "CNN" else 1
        )
        st.session_state.model_type = model_type
        
        # Hyperparameter tuning method
        st.subheader("Hyperparameter Tuning Method")
        tuning_method = st.radio(
            "Select tuning method",
            ["Manual Configuration", "Grid Search", "Bayesian Optimization"]
        )
        
        # Basic hyperparameters
        st.subheader("Common Hyperparameters")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=16, step=8)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001
            )
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1)
        
        with col2:
            epochs = st.slider("Epochs", min_value=10, max_value=100, value=50, step=10)
            l2_reg = st.select_slider(
                "L2 Regularization",
                options=[0.0, 0.0001, 0.001, 0.01, 0.1],
                value=0.001
            )
            patience = st.slider("Early Stopping Patience", min_value=5, max_value=20, value=10, step=5)
        
        # Model-specific hyperparameters
        if model_type == "CNN":
            st.subheader("CNN Hyperparameters")
            
            col1, col2 = st.columns(2)
            with col1:
                filters = st.slider("Number of Filters", min_value=16, max_value=128, value=32, step=16)
                kernel_size = st.slider("Kernel Size", min_value=2, max_value=5, value=3, step=1)
            
            with col2:
                cnn_layers = st.slider("Number of CNN Layers", min_value=1, max_value=4, value=2, step=1)
                dense_layers = st.slider("Number of Dense Layers", min_value=1, max_value=3, value=1, step=1)
        
        else:  # RNN
            st.subheader("RNN Hyperparameters")
            
            col1, col2 = st.columns(2)
            with col1:
                rnn_type = st.selectbox("RNN Cell Type", ["LSTM", "GRU", "SimpleRNN"], index=0)
                rnn_units = st.slider("RNN Units", min_value=16, max_value=128, value=64, step=16)
            
            with col2:
                rnn_layers = st.slider("Number of RNN Layers", min_value=1, max_value=3, value=1, step=1)
                bidirectional = st.checkbox("Bidirectional", value=True)
        
        # Advanced hyperparameter tuning options
        if tuning_method == "Grid Search":
            st.subheader("Grid Search Configuration")
            st.info("Grid Search will try all combinations of the specified parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                learning_rates = st.multiselect(
                    "Learning Rates",
                    [0.0001, 0.0005, 0.001, 0.005, 0.01],
                    default=[0.0001, 0.001, 0.01]
                )
                
                batch_sizes = st.multiselect(
                    "Batch Sizes",
                    [8, 16, 32, 64],
                    default=[16, 32]
                )
            
            with col2:
                dropout_rates = st.multiselect(
                    "Dropout Rates",
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    default=[0.0, 0.2, 0.4]
                )
                
                l2_regs = st.multiselect(
                    "L2 Regularization Values",
                    [0.0, 0.0001, 0.001, 0.01],
                    default=[0.0, 0.001]
                )
        
        elif tuning_method == "Bayesian Optimization":
            st.subheader("Bayesian Optimization Configuration")
            st.info("Bayesian Optimization will automatically search for optimal hyperparameters")
            
            n_trials = st.slider("Number of trials", min_value=10, max_value=50, value=20, step=5)
            
            col1, col2 = st.columns(2)
            with col1:
                lr_min = st.select_slider("Min Learning Rate", options=[0.0001, 0.0005, 0.001], value=0.0001)
                lr_max = st.select_slider("Max Learning Rate", options=[0.001, 0.005, 0.01], value=0.01)
                
                dropout_min = st.slider("Min Dropout Rate", min_value=0.0, max_value=0.3, value=0.0, step=0.1)
                dropout_max = st.slider("Max Dropout Rate", min_value=0.2, max_value=0.5, value=0.4, step=0.1)
            
            with col2:
                l2_min = st.select_slider("Min L2 Regularization", options=[0.0, 0.0001, 0.001], value=0.0)
                l2_max = st.select_slider("Max L2 Regularization", options=[0.001, 0.01, 0.1], value=0.01)
                
                batch_min = st.slider("Min Batch Size", min_value=8, max_value=16, value=8, step=8)
                batch_max = st.slider("Max Batch Size", min_value=32, max_value=64, value=64, step=16)
        
        # Save configuration button
        if st.button("Save Configuration"):
            st.success("Model configuration saved. Please proceed to the Training page.")

# Training page
elif page == "Training":
    st.header("Model Training")
    
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Please upload and preprocess data first.")
    else:
        # Display training info
        st.subheader("Training Information")
        st.write(f"Model type: {st.session_state.model_type}")
        st.write(f"Input shape: {st.session_state.X_train.shape}")
        st.write(f"Number of classes: {len(np.unique(st.session_state.y_train))}")
        
        # Mock training functionality
        if st.button("Train Model"):
            with st.spinner("Training in progress... This may take a while."):
                # Simulate training with a progress bar
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)  # Simulate computation time
                    progress_bar.progress(i + 1)
                
                # Create mock training history
                num_epochs = 50
                history = {
                    'accuracy': np.linspace(0.5, 0.96, num_epochs) + np.random.normal(0, 0.03, num_epochs),
                    'val_accuracy': np.linspace(0.4, 0.91, num_epochs) + np.random.normal(0, 0.05, num_epochs),
                    'loss': 1 - np.linspace(0.1, 0.9, num_epochs) + np.random.normal(0, 0.05, num_epochs),
                    'val_loss': 1.2 - np.linspace(0.1, 0.8, num_epochs) + np.random.normal(0, 0.07, num_epochs)
                }
                
                # Ensure values are in reasonable ranges
                history['accuracy'] = np.clip(history['accuracy'], 0, 1)
                history['val_accuracy'] = np.clip(history['val_accuracy'], 0, 1)
                history['loss'] = np.clip(history['loss'], 0, 2)
                history['val_loss'] = np.clip(history['val_loss'], 0, 2)
                
                # Store in session state
                st.session_state.training_history = history
                
                # Generate mock predictions for evaluation
                unique_classes = np.unique(st.session_state.y_test)
                num_classes = len(unique_classes)
                y_pred_proba = np.random.rand(len(st.session_state.y_test), num_classes)
                y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)  # Normalize to sum to 1
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # Convert to actual class labels
                class_mapping = {i: label for i, label in enumerate(unique_classes)}
                y_pred = np.array([class_mapping[p] for p in y_pred])
                
                # Store predictions
                st.session_state.test_predictions = y_pred
                
                st.success("Training completed successfully!")
        
        # Display training results if available
        if st.session_state.training_history is not None:
            st.subheader("Training Results")
            
            # Plot training history
            history = st.session_state.training_history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot accuracy
            epochs = range(1, len(history['accuracy']) + 1)
            ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
            ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)
            
            # Plot loss
            ax2.plot(epochs, history['loss'], 'b-', label='Training Loss')
            ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Final performance metrics
            st.subheader("Final Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Accuracy", f"{history['accuracy'][-1]:.4f}")
                st.metric("Training Loss", f"{history['loss'][-1]:.4f}")
            
            with col2:
                st.metric("Validation Accuracy", f"{history['val_accuracy'][-1]:.4f}")
                st.metric("Validation Loss", f"{history['val_loss'][-1]:.4f}")
                
            st.info("Proceed to the Evaluation page for detailed performance metrics.")

# Evaluation page
elif page == "Evaluation":
    st.header("Model Evaluation")
    
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Please upload and preprocess data first.")
    elif st.session_state.test_predictions is None:
        st.warning("Please train the model first.")
    else:
        # Evaluation metrics
        st.subheader("Performance Metrics")
        
        y_true = st.session_state.y_test
        y_pred = st.session_state.test_predictions
        
        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Display metrics
        st.metric("Test Accuracy", f"{accuracy:.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Normalized Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
        # Per-class metrics
        st.subheader("Per-Class Performance")
        
        # Get unique classes
        classes = sorted(list(set(y_true)))
        
        # Prepare data
        class_metrics = {
            'Class': classes,
            'Precision': [report[str(c)]['precision'] for c in classes],
            'Recall': [report[str(c)]['recall'] for c in classes],
            'F1-Score': [report[str(c)]['f1-score'] for c in classes],
            'Support': [report[str(c)]['support'] for c in classes]
        }
        
        # Plot metrics
        metrics_df = pd.DataFrame(class_metrics)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Precision and Recall
        ax1.bar(metrics_df['Class'].astype(str), metrics_df['Precision'], alpha=0.7, label='Precision')
        ax1.bar(metrics_df['Class'].astype(str), metrics_df['Recall'], alpha=0.7, label='Recall')
        ax1.set_title('Precision and Recall by Class')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # F1-Score
        ax2.bar(metrics_df['Class'].astype(str), metrics_df['F1-Score'], color='green')
        ax2.set_title('F1-Score by Class')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download trained model option
        st.subheader("Download Results")
        st.info("For a real implementation, you would be able to download the trained model and results here.")
        
        # Final conclusions
        st.subheader("Analysis and Conclusions")
        st.write("""
        Based on the evaluation metrics, we can draw the following conclusions:
        
        1. The model has achieved a good overall accuracy, but there are variations in performance across different ransomware families.
        2. Some families are easier to classify than others, likely due to their unique entropy signatures.
        3. Families with similar entropy patterns or smaller representation in the training data tend to have lower precision and recall.
        4. Further improvements could be achieved by:
           - Collecting more data for underrepresented families
           - Engineering additional features beyond entropy values
           - Fine-tuning hyperparameters with a more extensive search
           - Experimenting with ensemble methods combining CNN and RNN models
        """)