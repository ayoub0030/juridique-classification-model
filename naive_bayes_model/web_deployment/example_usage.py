from model_helper import NaiveBayesLegalClassifier

# Initialize the classifier
classifier = NaiveBayesLegalClassifier(model_dir='./model')

# Get model info
model_info = classifier.get_model_info()
print("Model Information:")
print(f"  Accuracy: {model_info['accuracy']:.4f}")
print(f"  Classes: {model_info['classes']}")
print(f"  Average inference time: {model_info['average_inference_time']:.2f} ms")

# Example document text
example_text = """
REGULATION (EU) No 1025/2012 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL
of 25 October 2012
on European standardisation, amending Council Directives 89/686/EEC and 93/15/EEC and Directives 94/9/EC,
94/25/EC, 95/16/EC, 97/23/EC, 98/34/EC, 2004/22/EC, 2007/23/EC, 2009/23/EC and 2009/105/EC of the
European Parliament and of the Council and repealing Council Decision 87/95/EEC and Decision
No 1673/2006/EC of the European Parliament and of the Council
"""

# Make a prediction
result = classifier.predict(example_text)

print("\nPrediction Results:")
print(f"  Predicted class: {result['prediction']}")
print("  Class probabilities:")
for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"    {cls}: {prob:.4f}")
print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
