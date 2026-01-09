import pandas as pd
import numpy as np
import math
import re
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("SPAM EMAIL DETECTION SYSTEM - LOGISTIC REGRESSION FROM SCRATCH")
print("=" * 70)

# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================

print("\n1. LOADING AND PREPARING DATA...")

# Load dataset
file_name = 'n_meisrishvili25_87421.csv'
try:
    df = pd.read_csv(file_name)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except:
    print(f"✗ Error: File '{file_name}' not found!")
    exit(1)

# Display dataset info
print(f"\n  Features: {list(df.columns)}")
print(f"  First 5 rows:")
print(df.head())

# Prepare features and target
X = df[['words', 'links', 'capital_words', 'spam_word_count']].values
y = df['is_spam'].values.reshape(-1, 1)

print(f"\n  Dataset statistics:")
print(f"    X shape: {X.shape}")
print(f"    y shape: {y.shape}")
print(f"    Spam emails: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")
print(f"    Legitimate emails: {len(y) - y.sum()} ({(len(y) - y.sum()) / len(y) * 100:.1f}%)")

# ============================================
# 2. MANUAL TRAIN-TEST SPLIT (70/30)
# ============================================

print("\n2. SPLITTING DATA (70% train, 30% test)...")

# Set random seed for reproducibility
np.random.seed(42)

# Create indices and shuffle
indices = np.arange(len(X))
np.random.shuffle(indices)

# Calculate split point
split_idx = int(0.7 * len(X))

# Split data
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set: {X_test.shape[0]} samples")
print(f"  Training - Spam: {y_train.sum()}, Legitimate: {len(y_train) - y_train.sum()}")
print(f"  Testing - Spam: {y_test.sum()}, Legitimate: {len(y_test) - y_test.sum()}")

# ============================================
# 3. LOGISTIC REGRESSION FROM SCRATCH
# ============================================

print("\n3. IMPLEMENTING LOGISTIC REGRESSION FROM SCRATCH...")


def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))


def add_bias(X):
    """Add bias term (intercept) to features"""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def logistic_regression_fit(X, y, learning_rate=0.01, epochs=1000):
    """
    Train logistic regression model using gradient descent
    """
    # Add bias term
    X_bias = add_bias(X)

    # Initialize weights (including bias)
    n_features = X_bias.shape[1]
    weights = np.zeros((n_features, 1))

    # Gradient descent
    m = len(X)
    losses = []

    for epoch in range(epochs):
        # Forward pass
        z = np.dot(X_bias, weights)
        predictions = sigmoid(z)

        # Calculate loss (binary cross-entropy)
        epsilon = 1e-15  # Small value to avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        losses.append(loss)

        # Calculate gradient
        gradient = np.dot(X_bias.T, (predictions - y)) / m

        # Update weights
        weights -= learning_rate * gradient

        # Print progress every 100 epochs
        if epoch % 200 == 0:
            print(f"    Epoch {epoch:4d}: Loss = {loss:.6f}")

    return weights, losses


def logistic_regression_predict(X, weights):
    """Make predictions using trained weights"""
    X_bias = add_bias(X)
    probabilities = sigmoid(np.dot(X_bias, weights))
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, probabilities


print("  Training model with gradient descent...")
weights, losses = logistic_regression_fit(X_train, y_train, learning_rate=0.1, epochs=1000)

print(f"\n  Training completed!")
print(f"  Final weights:")
feature_names = ['bias', 'words', 'links', 'capital_words', 'spam_word_count']
for name, weight in zip(feature_names, weights.flatten()):
    print(f"    {name:15s}: {weight:.6f}")

# ============================================
# 4. MODEL EVALUATION
# ============================================

print("\n4. EVALUATING MODEL ON TEST DATA...")

# Make predictions
y_pred, y_prob = logistic_regression_predict(X_test, weights)


# Calculate confusion matrix manually
def confusion_matrix_manual(y_true, y_pred):
    """Calculate confusion matrix without sklearn"""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


TP, TN, FP, FN = confusion_matrix_manual(y_test, y_pred)

print(f"\n  CONFUSION MATRIX:")
print(f"                  Predicted")
print(f"                  Legit    Spam")
print(f"  Actual Legit    {TN:5d}    {FP:5d}")
print(f"  Actual Spam     {FN:5d}    {TP:5d}")

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

print(f"\n  PERFORMANCE METRICS:")
print(f"    Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"    Precision:   {precision:.4f} ({precision * 100:.2f}%)")
print(f"    Recall:      {recall:.4f} ({recall * 100:.2f}%)")
print(f"    F1-Score:    {f1_score:.4f} ({f1_score * 100:.2f}%)")
print(f"    Specificity: {specificity:.4f} ({specificity * 100:.2f}%)")

# ============================================
# 5. FEATURE EXTRACTION FOR NEW EMAILS
# ============================================

print("\n5. FEATURE EXTRACTION FUNCTION...")


def extract_features_from_email(email_text):
    """
    Extract the 4 features from email text
    """
    features = {}

    # 1. Word count
    words = email_text.split()
    features['words'] = len(words)

    # 2. Link count
    link_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    links = re.findall(link_pattern, email_text)
    features['links'] = len(links)

    # 3. Capitalized words (ALL CAPS, length > 1)
    capital_words = [word for word in words if word.isupper() and len(word) > 1]
    features['capital_words'] = len(capital_words)

    # 4. Spam word count
    spam_words = [
        'free', 'win', 'winner', 'prize', 'cash', 'money', 'million',
        'click', 'buy', 'order', 'offer', 'discount', 'save', 'sale',
        'urgent', 'immediate', 'limited', 'guaranteed', 'winner', 'selected',
        'congratulations', 'risk', 'free', 'trial', 'act', 'now', 'today',
        'special', 'promotion', 'exclusive', 'claim', 'call', 'subscribe'
    ]

    text_lower = email_text.lower()
    spam_count = 0
    for word in spam_words:
        spam_count += text_lower.count(word)
    features['spam_word_count'] = min(spam_count, 10)  # Cap at 10 like dataset

    return features


def classify_email_manual(email_text, weights):
    """
    Classify email using manual logistic regression
    """
    # Extract features
    features = extract_features_from_email(email_text)

    # Create feature array in correct order
    X_new = np.array([[features['words'], features['links'],
                       features['capital_words'], features['spam_word_count']]])

    # Make prediction
    prediction, probability = logistic_regression_predict(X_new, weights)

    return {
        'prediction': 'SPAM' if prediction[0][0] == 1 else 'LEGITIMATE',
        'prediction_numeric': prediction[0][0],
        'spam_probability': probability[0][0] * 100,
        'legitimate_probability': (1 - probability[0][0]) * 100,
        'features': features
    }


# ============================================
# 6. TEST WITH MANUAL EMAILS
# ============================================

print("\n" + "=" * 70)
print("MANUAL EMAIL CLASSIFICATION TEST")
print("=" * 70)

# Manual Spam Email
manual_spam = """
URGENT! WIN $10,000 CASH PRIZE - FREE ENTRY!

Dear Valued Customer,

CONGRATULATIONS! You have been SELECTED as a potential WINNER 
of our $10,000 GRAND PRIZE! This is a LIMITED TIME OFFER!

To CLAIM your FREE REWARD, CLICK this link NOW: 
http://win-prize-now.xyz/claim?user=winner123
http://special-offer.com/bonus
http://get-money-fast.net

ACT IMMEDIATELY! This offer expires in 24 HOURS!!!

Best regards,
Prize Distribution Team
"""

# Manual Legitimate Email
manual_legit = """
Subject: Project Update and Team Meeting Schedule

Hello Team,

I hope you are having a productive week. 

Attached please find the updated project timeline for Q1 2026. 
We will discuss this in our weekly meeting on Friday at 10:00 AM 
in the main conference room.

Please review the document and come prepared with any questions 
or suggestions you may have.

The agenda will include:
1. Q1 milestones review
2. Resource allocation update
3. Risk assessment discussion
4. Next steps planning

Best regards,
Alex Johnson
Project Lead
IT Department
Acme Corporation
"""

print("\nA. TESTING MANUALLY CREATED SPAM EMAIL:")
print("-" * 50)
print(manual_spam[:150] + "...")

spam_result = classify_email_manual(manual_spam, weights)
print(f"\n  Extracted Features:")
for feat, val in spam_result['features'].items():
    print(f"    {feat:15s}: {val}")
print(f"\n  Classification Result:")
print(f"    Prediction: {spam_result['prediction']}")
print(f"    Spam probability: {spam_result['spam_probability']:.2f}%")

print("\nB. TESTING MANUALLY CREATED LEGITIMATE EMAIL:")
print("-" * 50)
print(manual_legit[:150] + "...")

legit_result = classify_email_manual(manual_legit, weights)
print(f"\n  Extracted Features:")
for feat, val in legit_result['features'].items():
    print(f"    {feat:15s}: {val}")
print(f"\n  Classification Result:")
print(f"    Prediction: {legit_result['prediction']}")
print(f"    Spam probability: {legit_result['spam_probability']:.2f}%")

# ============================================
# 7. SAVE RESULTS
# ============================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save model weights
weights_df = pd.DataFrame({
    'feature': feature_names,
    'weight': weights.flatten(),
    'absolute_weight': np.abs(weights.flatten())
})
weights_df.to_csv('model_weights_manual.csv', index=False)
print(f"✓ Model weights saved as 'model_weights_manual.csv'")

# Save test predictions
test_results = pd.DataFrame({
    'words': X_test[:, 0],
    'links': X_test[:, 1],
    'capital_words': X_test[:, 2],
    'spam_word_count': X_test[:, 3],
    'actual_is_spam': y_test.flatten(),
    'predicted_is_spam': y_pred.flatten(),
    'spam_probability': y_prob.flatten(),
    'correct': y_test.flatten() == y_pred.flatten()
})
test_results.to_csv('test_predictions_manual.csv', index=False)
print(f"✓ Test predictions saved as 'test_predictions_manual.csv'")

# Save metrics
metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'TN', 'FP', 'FN', 'TP'],
    'Value': [accuracy, precision, recall, f1_score, specificity, TN, FP, FN, TP]
})
metrics.to_csv('model_metrics_manual.csv', index=False)
print(f"✓ Metrics saved as 'model_metrics_manual.csv'")

# Save email examples
examples = pd.DataFrame([{
    'email_type': 'manual_spam',
    'words': spam_result['features']['words'],
    'links': spam_result['features']['links'],
    'capital_words': spam_result['features']['capital_words'],
    'spam_word_count': spam_result['features']['spam_word_count'],
    'prediction': spam_result['prediction'],
    'spam_probability': spam_result['spam_probability']
}, {
    'email_type': 'manual_legit',
    'words': legit_result['features']['words'],
    'links': legit_result['features']['links'],
    'capital_words': legit_result['features']['capital_words'],
    'spam_word_count': legit_result['features']['spam_word_count'],
    'prediction': legit_result['prediction'],
    'spam_probability': legit_result['spam_probability']
}])
examples.to_csv('example_classifications_manual.csv', index=False)
print(f"✓ Example classifications saved as 'example_classifications_manual.csv'")

# ============================================
# 8. VISUALIZATION FUNCTIONS
# ============================================

print("\n8. GENERATING VISUALIZATIONS...")

try:
    import matplotlib.pyplot as plt

    # Visualization 1: Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title('Training Loss (Binary Cross-Entropy)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: training_loss.png")

    # Visualization 2: Feature Weights
    plt.figure(figsize=(10, 6))
    colors = ['red' if w > 0 else 'green' for w in weights.flatten()[1:]]  # Skip bias
    bars = plt.bar(feature_names[1:], np.abs(weights.flatten()[1:]), color=colors, edgecolor='black')

    plt.title('Feature Importance (Absolute Weight Values)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Absolute Weight Value', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    # Add weight values on bars
    for bar, weight in zip(bars, weights.flatten()[1:]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{weight:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('feature_weights_manual.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: feature_weights_manual.png")

    # Visualization 3: Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm_values = [[TN, FP], [FN, TP]]
    cm_labels = [['TN', 'FP'], ['FN', 'TP']]

    plt.imshow(cm_values, cmap='Reds', interpolation='nearest')
    plt.colorbar(label='Count')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks([0, 1], ['Legit', 'Spam'])
    plt.yticks([0, 1], ['Legit', 'Spam'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm_labels[i][j]}\n{cm_values[i][j]}',
                     ha='center', va='center',
                     color='white' if cm_values[i][j] > max(cm_values[0] + cm_values[1]) / 2 else 'black',
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('confusion_matrix_manual.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrix_manual.png")

except ImportError:
    print("  Note: matplotlib not installed. Skipping visualizations.")
    print("  Run: pip install matplotlib")

print("\n" + "=" * 70)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nSUMMARY:")
print(f"• Dataset: {len(df)} emails")
print(f"• Model: Logistic Regression (manual implementation)")
print(f"• Training iterations: 1000 epochs")
print(f"• Accuracy on test data: {accuracy * 100:.2f}%")
print(f"• Manual spam email classified as: {spam_result['prediction']}")
print(f"• Manual legitimate email classified as: {legit_result['prediction']}")
print(f"\nAll required outputs saved to CSV files.")