# BEGINNER-FRIENDLY BLUR DETECTION MODEL WITH COMPREHENSIVE EVALUATION
# This code will help you detect if an image is blurred or sharp with detailed performance metrics!

# Step 1: Import all the libraries we need
import numpy as np                    # For working with arrays/numbers
import cv2                           # For image processing
import os                           # For file operations
import matplotlib.pyplot as plt     # For showing images and graphs
import seaborn as sns               # For beautiful statistical plots
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, roc_auc_score, precision_recall_curve)  # All evaluation metrics
import tensorflow as tf             # Main AI library
from tensorflow import keras        # High-level AI library
from tensorflow.keras import layers # For building neural network layers
import kagglehub                   # For downloading dataset from Kaggle
from tqdm import tqdm              # For progress bars
import warnings
warnings.filterwarnings('ignore')   # Hide minor warnings for cleaner output

print("✅ All libraries imported successfully!")

class SimpleBlurDetector:
    """
    A simple class to detect if images are blurred or sharp
    Perfect for beginners learning AI/ML with comprehensive evaluation!
    """

    def __init__(self):
        # Set image size - all images will be resized to this
        self.image_width = 128   # Making it smaller for faster training
        self.image_height = 128
        self.model = None        # Our AI model (empty for now)
        self.X_test = None       # Store test data for evaluation
        self.y_test = None       # Store test labels for evaluation

        print("🎯 Blur Detector initialized!")
        print(f"📏 Images will be resized to {self.image_width}x{self.image_height} pixels")

    def download_dataset(self):
        """
        Step 1: Download the dataset from Kaggle
        This dataset has blurred and sharp images for training
        """
        print("📥 Downloading dataset from Kaggle...")
        print("(This might take a few minutes)")

        try:
            path = kagglehub.dataset_download("kwentar/blur-dataset")
            print(f"✅ Dataset downloaded successfully!")
            print(f"📂 Dataset location: {path}")
            return path
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("💡 Make sure you have internet connection and kagglehub installed")
            return None

    def load_single_image(self, image_path):
        """
        Load and prepare a single image for our AI model
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Convert color format (OpenCV uses BGR, we want RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to our standard size
            image = cv2.resize(image, (self.image_width, self.image_height))

            # Convert to numbers between 0 and 1 (normalization)
            image = image.astype(np.float32) / 255.0

            return image
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None

    def load_dataset(self, dataset_path):
        """
        Step 2: Load all images from the dataset
        """
        print("📚 Loading dataset...")

        images = []      # List to store all images
        labels = []      # List to store labels (0=sharp, 1=blur)

        # Path to blurred images
        blur_folder = os.path.join(dataset_path, 'defocused_blurred')
        # Path to sharp images
        sharp_folder = os.path.join(dataset_path, 'sharp')

        # Load blurred images
        if os.path.exists(blur_folder):
            print("🔍 Loading blurred images...")
            blur_files = [f for f in os.listdir(blur_folder)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for filename in tqdm(blur_files[:1000], desc="Blurred images"):  # Limit to 1000 for speed
                image_path = os.path.join(blur_folder, filename)
                image = self.load_single_image(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(1)  # 1 means "blurred"

        # Load sharp images
        if os.path.exists(sharp_folder):
            print("✨ Loading sharp images...")
            sharp_files = [f for f in os.listdir(sharp_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for filename in tqdm(sharp_files[:1000], desc="Sharp images"):  # Limit to 1000 for speed
                image_path = os.path.join(sharp_folder, filename)
                image = self.load_single_image(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(0)  # 0 means "sharp"

        # Convert lists to numpy arrays (format needed for AI)
        X = np.array(images)
        y = np.array(labels)

        print(f"✅ Dataset loaded!")
        print(f"📊 Total images: {len(X)}")
        print(f"🔢 Sharp images: {np.sum(y == 0)}")
        print(f"🌀 Blurred images: {np.sum(y == 1)}")

        return X, y

    def create_model(self):
        """
        Step 3: Create our AI model
        This is a Convolutional Neural Network (CNN) - great for images!
        """
        print("🧠 Creating AI model...")

        # Create a sequential model (layers stacked one after another)
        model = keras.Sequential([

            # Input layer - where images enter the model
            layers.Input(shape=(self.image_width, self.image_height, 3)),  # 3 for RGB colors

            # First layer: Look for simple patterns (edges, colors)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),  # Make image smaller
            layers.Dropout(0.25),         # Prevent overfitting

            # Second layer: Look for more complex patterns
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third layer: Even more complex patterns
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Flatten the image into a long list of numbers
            layers.GlobalAveragePooling2D(),

            # Dense layers: Make final decisions
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),

            # Output layer: Final answer (sharp or blur)
            layers.Dense(1, activation='sigmoid')  # Sigmoid gives probability between 0 and 1
        ])

        # Configure the model for training
        model.compile(
            optimizer='adam',           # Algorithm for learning
            loss='binary_crossentropy', # How to measure mistakes
            metrics=['accuracy']        # Track accuracy during training
        )

        self.model = model

        print("✅ Model created successfully!")
        print("📋 Model summary:")
        self.model.summary()

        return model

    def train_model(self, X, y, epochs=20):
        """
        Step 4: Train the model to recognize blurred vs sharp images
        """
        print("🎓 Starting training...")
        print("(This will take several minutes)")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test

        print(f"📚 Training images: {len(X_train)}")
        print(f"🧪 Testing images: {len(X_test)}")

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,           # Number of times to see all training data
            batch_size=32,          # How many images to process at once
            validation_data=(X_test, y_test),  # Test data to check progress
            verbose=1               # Show progress
        )

        print("✅ Training completed!")

        # Comprehensive evaluation
        self.comprehensive_evaluation()

        # Show training progress
        self.plot_training_history(history)

        return history

    def comprehensive_evaluation(self):
        """
        NEW: Comprehensive model evaluation with all important metrics
        """
        if self.model is None or self.X_test is None:
            print("❌ Model not trained yet!")
            return

        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)

        # Get predictions
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = self.y_test

        # 1. BASIC METRICS
        print("\n📊 BASIC PERFORMANCE METRICS:")
        print("-" * 40)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
        print(f"🔍 Precision: {precision:.4f} ({precision:.1%})")
        print(f"📡 Recall:    {recall:.4f} ({recall:.1%})")
        print(f"⚖️  F1-Score:  {f1:.4f} ({f1:.1%})")

        # 2. CONFUSION MATRIX
        print("\n📋 CONFUSION MATRIX:")
        print("-" * 40)
        cm = confusion_matrix(y_true, y_pred)

        # Text representation
        print("        Predicted")
        print("      Sharp  Blur")
        print(f"Sharp   {cm[0,0]:3d}   {cm[0,1]:3d}")
        print(f"Blur    {cm[1,0]:3d}   {cm[1,1]:3d}")

        # Calculate percentages
        tn, fp, fn, tp = cm.ravel()
        print(f"\n📈 True Positives (Correct Blur):  {tp}")
        print(f"📈 True Negatives (Correct Sharp): {tn}")
        print(f"📉 False Positives (Wrong Blur):   {fp}")
        print(f"📉 False Negatives (Wrong Sharp):  {fn}")

        # 3. DETAILED CLASSIFICATION REPORT
        print("\n📋 DETAILED CLASSIFICATION REPORT:")
        print("-" * 40)
        report = classification_report(y_true, y_pred,
                                     target_names=['Sharp', 'Blur'],
                                     digits=4)
        print(report)

        # 4. ROC AUC SCORE
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"🎪 ROC AUC Score: {roc_auc:.4f} ({roc_auc:.1%})")

        # 5. CREATE VISUAL PLOTS
        self.plot_evaluation_metrics(y_true, y_pred, y_pred_proba)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }

    def plot_evaluation_metrics(self, y_true, y_pred, y_pred_proba):
        """
        NEW: Create comprehensive visualization of model performance
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Sharp', 'Blur'],
                   yticklabels=['Sharp', 'Blur'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        axes[0, 2].plot(recall, precision, color='purple', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.2f})')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Prediction Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7,
                       label='Sharp Images', color='green', density=True)
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7,
                       label='Blur Images', color='red', density=True)
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Prediction Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Class-wise Performance Bar Chart
        metrics = ['Precision', 'Recall', 'F1-Score']
        sharp_scores = [
            precision_score(y_true, y_pred, pos_label=0),
            recall_score(y_true, y_pred, pos_label=0),
            f1_score(y_true, y_pred, pos_label=0)
        ]
        blur_scores = [
            precision_score(y_true, y_pred, pos_label=1),
            recall_score(y_true, y_pred, pos_label=1),
            f1_score(y_true, y_pred, pos_label=1)
        ]

        x = np.arange(len(metrics))
        width = 0.35

        axes[1, 1].bar(x - width/2, sharp_scores, width, label='Sharp', color='green', alpha=0.7)
        axes[1, 1].bar(x + width/2, blur_scores, width, label='Blur', color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Class-wise Performance', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)

        # 6. Model Confidence Analysis
        correct_preds = (y_pred == y_true)
        correct_conf = y_pred_proba[correct_preds]
        wrong_conf = y_pred_proba[~correct_preds]

        # Adjust confidence for sharp predictions
        correct_conf_adj = np.where(y_pred[correct_preds] == 1, correct_conf, 1 - correct_conf)
        wrong_conf_adj = np.where(y_pred[~correct_preds] == 1, wrong_conf, 1 - wrong_conf)

        axes[1, 2].hist(correct_conf_adj.flatten(), bins=20, alpha=0.7,
                       label='Correct Predictions', color='green', density=True)
        axes[1, 2].hist(wrong_conf_adj.flatten(), bins=20, alpha=0.7,
                       label='Wrong Predictions', color='red', density=True)
        axes[1, 2].set_xlabel('Confidence Level')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].set_title('Model Confidence Analysis', fontsize=14, fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_training_history(self, history):
        """
        Show how well the model learned during training
        """
        plt.figure(figsize=(15, 5))

        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
        plt.title('Model Accuracy During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch (Training Round)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch (Training Round)')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Learning rate (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], color='orange', linewidth=2)
            plt.title('Learning Rate', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
        else:
            # Show the difference between training and validation accuracy
            acc_diff = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
            plt.plot(acc_diff, color='purple', linewidth=2)
            plt.title('Overfitting Monitor\n(Training - Validation Accuracy)', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy Difference')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_image(self, image_path):
        """
        Step 5: Use the trained model to predict if an image is blurred
        """
        if self.model is None:
            print("❌ Model not trained yet! Please train the model first.")
            return None

        print(f"🔍 Analyzing image: {image_path}")

        # Load and prepare the image
        image = self.load_single_image(image_path)
        if image is None:
            print("❌ Could not load the image. Please check the file path.")
            return None

        # Add batch dimension (model expects multiple images)
        image_batch = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = self.model.predict(image_batch, verbose=0)
        probability = float(prediction[0][0])  # Get the probability

        # Interpret the result
        if probability > 0.5:
            result = "BLURRED"
            confidence = probability
        else:
            result = "SHARP"
            confidence = 1 - probability

        # Show the image with enhanced information
        plt.figure(figsize=(10, 8))
        plt.imshow(image)

        # Enhanced title with more information
        title_color = 'red' if result == "BLURRED" else 'green'
        confidence_bar = "█" * int(confidence * 20)  # Visual confidence bar

        plt.title(f'Prediction: {result}\n'
                 f'Confidence: {confidence:.1%} {confidence_bar}\n'
                 f'Raw Probability: {probability:.4f}',
                 fontsize=16, fontweight='bold', color=title_color)
        plt.axis('off')

        # Add confidence interpretation
        if confidence > 0.9:
            conf_text = "Very Confident ✅"
        elif confidence > 0.7:
            conf_text = "Confident 👍"
        elif confidence > 0.6:
            conf_text = "Somewhat Confident 🤔"
        else:
            conf_text = "Low Confidence ⚠️"

        plt.figtext(0.5, 0.02, conf_text, ha='center', fontsize=14,
                   style='italic', color=title_color)

        plt.tight_layout()
        plt.show()

        print(f"🎯 Result: {result}")
        print(f"📊 Confidence: {confidence:.1%}")
        print(f"🔢 Raw Probability: {probability:.4f}")
        print(f"💭 Interpretation: {conf_text}")

        return {
            'prediction': result,
            'confidence': confidence,
            'probability': probability,
            'interpretation': conf_text
        }

    def save_model(self, filename='my_blur_detector.h5'):
        """
        Save the trained model so you don't have to train again
        """
        if self.model is None:
            print("❌ No model to save!")
            return

        self.model.save(filename)
        print(f"💾 Model saved as '{filename}'")

    def load_model(self, filename):
        """
        Load a previously saved model
        """
        try:
            self.model = keras.models.load_model(filename)
            print(f"✅ Model loaded from '{filename}'")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def model_summary_report(self):
        """
        NEW: Generate a comprehensive summary report of the model
        """
        if self.model is None:
            print("❌ Model not available!")
            return

        print("\n" + "="*60)
        print("📋 COMPREHENSIVE MODEL SUMMARY REPORT")
        print("="*60)

        # Model architecture info
        total_params = self.model.count_params()
        trainable_params = sum([tf.reduce_prod(var.shape) for var in self.model.trainable_variables])

        print(f"\n🏗️  MODEL ARCHITECTURE:")
        print(f"   • Total Parameters: {total_params:,}")
        print(f"   • Trainable Parameters: {trainable_params:,}")
        print(f"   • Input Shape: {self.image_width}x{self.image_height}x3")
        print(f"   • Output: Binary Classification (Sharp/Blur)")

        # Performance summary
        if self.X_test is not None and self.y_test is not None:
            y_pred_proba = self.model.predict(self.X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            print(f"\n🎯 PERFORMANCE SUMMARY:")
            print(f"   • Overall Accuracy: {accuracy:.1%}")
            print(f"   • Precision: {precision:.1%}")
            print(f"   • Recall: {recall:.1%}")
            print(f"   • F1-Score: {f1:.1%}")

            # Performance interpretation
            if accuracy > 0.95:
                performance = "Excellent 🌟"
            elif accuracy > 0.90:
                performance = "Very Good 👍"
            elif accuracy > 0.85:
                performance = "Good ✅"
            elif accuracy > 0.75:
                performance = "Fair 🤔"
            else:
                performance = "Needs Improvement ⚠️"

            print(f"   • Overall Performance: {performance}")

def main():
    """
    MAIN PROGRAM - This is where everything happens!
    """
    print("🚀 WELCOME TO ENHANCED BLUR DETECTION WITH FULL EVALUATION!")
    print("=" * 70)
    print("This program will:")
    print("1. Download a dataset of blurred and sharp images")
    print("2. Train an AI model to recognize the difference")
    print("3. Provide comprehensive evaluation metrics")
    print("4. Let you test it on your own images!")
    print("=" * 70)

    # Create our blur detector
    detector = SimpleBlurDetector()

    # Step 1: Download dataset
    print("\n" + "="*20 + " STEP 1: DOWNLOAD DATA " + "="*20)
    dataset_path = detector.download_dataset()
    if dataset_path is None:
        print("❌ Cannot continue without dataset")
        return

    # Step 2: Load dataset
    print("\n" + "="*20 + " STEP 2: LOAD IMAGES " + "="*21)
    X, y = detector.load_dataset(dataset_path)

    # Step 3: Create model
    print("\n" + "="*20 + " STEP 3: CREATE MODEL " + "="*20)
    detector.create_model()

    # Step 4: Train model
    print("\n" + "="*20 + " STEP 4: TRAIN MODEL " + "="*21)
    detector.train_model(X, y, epochs=15)  # Using fewer epochs for faster training

    # Step 5: Generate comprehensive report
    print("\n" + "="*20 + " STEP 5: GENERATE REPORT " + "="*18)
    detector.model_summary_report()

    # Step 6: Save model
    print("\n" + "="*20 + " STEP 6: SAVE MODEL " + "="*22)
    detector.save_model('enhanced_blur_detector_model.h5')

    # Step 7: Test with user images
    print("\n" + "="*20 + " STEP 7: TEST YOUR IMAGES " + "="*17)
    print("🎉 Training complete! Now you can test your own images!")

    while True:
        print("\n" + "-"*50)
        user_input = input("📸 Enter the path to your image (or 'quit' to exit): ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye! Thanks for using the enhanced blur detector!")
            break

        if user_input == "":
            print("Please enter a valid image path!")
            continue

        # Test the image
        result = detector.predict_image(user_input)

        if result:
            print(f"\n✨ FINAL RESULT: Your image is {result['prediction']}")
            print(f"🎯 Confidence: {result['confidence']:.1%}")
            print(f"💭 Model says: {result['interpretation']}")

if __name__ == "__main__":
    main()
